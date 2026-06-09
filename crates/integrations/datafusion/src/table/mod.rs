// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Iceberg table providers for DataFusion.
//!
//! This module provides two table provider implementations:
//!
//! - [`IcebergTableProvider`]: Catalog-backed provider with automatic metadata refresh.
//!   Use for write operations and when you need to see the latest table state.
//!
//! - [`IcebergStaticTableProvider`]: Static provider for read-only access to a specific
//!   table snapshot. Use for consistent analytical queries or time-travel scenarios.

mod bucketing;
pub mod metadata_table;
pub mod table_provider_factory;

use std::any::Any;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::catalog::Session;
use datafusion::common::DataFusionError;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning};
use futures::TryStreamExt;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::inspect::MetadataTableType;
use iceberg::scan::FileScanTask;
use iceberg::spec::TableProperties;
use iceberg::table::Table;
use iceberg::{Catalog, Error, ErrorKind, NamespaceIdent, Result, TableIdent};
use metadata_table::IcebergMetadataTableProvider;

use crate::error::to_datafusion_error;
use crate::physical_plan::commit::IcebergCommitExec;
use crate::physical_plan::expr_to_predicate::convert_filters_to_predicate;
use crate::physical_plan::project::project_with_partition;
use crate::physical_plan::repartition::repartition;
use crate::physical_plan::scan::IcebergTableScan;
use crate::physical_plan::sort::sort_by_partition;
use crate::physical_plan::write::IcebergWriteExec;
use crate::{IcebergScanConfig, PartitioningOverride};

/// Catalog-backed table provider with automatic metadata refresh.
///
/// This provider loads fresh table metadata from the catalog on every scan and write
/// operation, ensuring you always see the latest table state. Use this when you need
/// write operations or want to see the most up-to-date data.
///
/// For read-only access to a specific snapshot without catalog overhead, use
/// [`IcebergStaticTableProvider`] instead.
#[derive(Debug, Clone)]
pub struct IcebergTableProvider {
    /// The catalog that manages this table
    catalog: Arc<dyn Catalog>,
    /// The table identifier (namespace + name)
    table_ident: TableIdent,
    /// A reference-counted arrow `Schema` (cached at construction)
    schema: ArrowSchemaRef,
}

impl IcebergTableProvider {
    /// Creates a new catalog-backed table provider.
    ///
    /// Loads the table once to get the initial schema, then stores the catalog
    /// reference for future metadata refreshes on each operation.
    pub(crate) async fn try_new(
        catalog: Arc<dyn Catalog>,
        namespace: NamespaceIdent,
        name: impl Into<String>,
    ) -> Result<Self> {
        let table_ident = TableIdent::new(namespace, name.into());

        let table = catalog.load_table(&table_ident).await?;
        let schema = Arc::new(schema_to_arrow_schema(table.metadata().current_schema())?);

        Ok(IcebergTableProvider {
            catalog,
            table_ident,
            schema,
        })
    }

    pub(crate) async fn metadata_table(
        &self,
        r#type: MetadataTableType,
    ) -> Result<IcebergMetadataTableProvider> {
        let table = self.catalog.load_table(&self.table_ident).await?;
        Ok(IcebergMetadataTableProvider { table, r#type })
    }
}

#[async_trait]
impl TableProvider for IcebergTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Second load: fetch the latest snapshot so scans always reflect current table state.
        let table = self
            .catalog
            .load_table(&self.table_ident)
            .await
            .map_err(to_datafusion_error)?;

        // Build a TableScan mirroring the inputs we'll hand to IcebergTableScan,
        // so plan_files() uses the same projection/filters the scan will replay in execute().
        let col_names = projection.map(|indices| {
            indices
                .iter()
                .map(|&i| self.schema.field(i).name().clone())
                .collect::<Vec<_>>()
        });

        let predicate = convert_filters_to_predicate(filters);

        let mut builder = table.scan();
        builder = match col_names {
            Some(names) => builder.select(names),
            None => builder.select_all(),
        };
        if let Some(pred) = predicate {
            builder = builder.with_filter(pred);
        }

        let tasks: Vec<FileScanTask> = builder
            .build()
            .map_err(to_datafusion_error)?
            .plan_files()
            .await
            .map_err(to_datafusion_error)?
            .try_collect::<Vec<_>>()
            .await
            .map_err(to_datafusion_error)?;

        // Output schema after projection: column indices in `Hash` exprs and any
        // Arrow array we hash must reference this schema, not the full table schema.
        let output_schema = match projection {
            None => self.schema.clone(),
            Some(p) => Arc::new(self.schema.project(p).map_err(|e| {
                to_datafusion_error(Error::new(ErrorKind::DataInvalid, e.to_string()))
            })?),
        };

        let target_partitions = state.config().target_partitions();
        // Always produce at least 1 partition so that DataFusion can schedule
        // the plan normally and callers can safely call execute(0). An empty
        // bucket simply yields an empty record-batch stream.
        let n_partitions = target_partitions.min(tasks.len()).max(1);

        // `keys` is `Some` iff the table's default spec is hash-declarable:
        // either pure-identity (or mixed identity+bucket, in which case only
        // the identity columns become the key) or single-column pure-bucket.
        // Any other shape (spec evolution, missing source column, mixed
        // bucket+other transform, multi-column pure-bucket, unsupported
        // identity dtype) collapses to `None`, which forces
        // `UnknownPartitioning` regardless of bucketing strategy.
        let keys = bucketing::compute_partition_keys(&table, &output_schema);

        let partitioning_enabled = resolve_partitioning_enabled(
            keys.as_ref().map(|keys| keys.kind()),
            state.config_options().extensions.get::<IcebergScanConfig>(),
            table.metadata().properties(),
        )?;

        let (buckets, partitioning) = match (partitioning_enabled, keys.as_ref()) {
            (true, Some(keys)) => {
                let exprs = keys.column_exprs();
                let (buckets, all_had_full_key) =
                    bucketing::bucket_tasks(tasks, n_partitions, Some(keys));
                let partitioning = if all_had_full_key && n_partitions > 0 {
                    Partitioning::Hash(exprs, n_partitions)
                } else {
                    Partitioning::UnknownPartitioning(n_partitions)
                };
                (buckets, partitioning)
            }
            _ => {
                let (buckets, _) = bucketing::bucket_tasks(tasks, n_partitions, None);
                (buckets, Partitioning::UnknownPartitioning(n_partitions))
            }
        };

        Ok(Arc::new(IcebergTableScan::new_with_tasks(
            table,
            None, // Always use current snapshot for catalog-backed provider
            self.schema.clone(),
            projection,
            filters,
            limit,
            buckets,
            partitioning,
        )))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // Push down all filters, as a single source of truth, the scanner will drop the filters which couldn't be push down
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn insert_into(
        &self,
        state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        _insert_op: InsertOp,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let table = self
            .catalog
            .load_table(&self.table_ident)
            .await
            .map_err(to_datafusion_error)?;

        let partition_spec = table.metadata().default_partition_spec();

        // Step 1: Project partition values for partitioned tables
        let plan_with_partition = if !partition_spec.is_unpartitioned() {
            project_with_partition(input, &table)?
        } else {
            input
        };

        // Step 2: Repartition for parallel processing
        let target_partitions =
            NonZeroUsize::new(state.config().target_partitions()).ok_or_else(|| {
                DataFusionError::Configuration(
                    "target_partitions must be greater than 0".to_string(),
                )
            })?;

        let repartitioned_plan =
            repartition(plan_with_partition, table.metadata_ref(), target_partitions)?;

        let fanout_enabled = table
            .metadata()
            .properties()
            .get(TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED)
            .map(|value| {
                value
                    .parse::<bool>()
                    .map_err(|e| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Invalid value for {}, expected 'true' or 'false'",
                                TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED
                            ),
                        )
                        .with_source(e)
                    })
                    .map_err(to_datafusion_error)
            })
            .transpose()?
            .unwrap_or(TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED_DEFAULT);

        let write_input = if fanout_enabled {
            repartitioned_plan
        } else {
            sort_by_partition(repartitioned_plan)?
        };

        let write_plan = Arc::new(IcebergWriteExec::new(
            table.clone(),
            write_input,
            self.schema.clone(),
        ));

        // Merge the outputs of write_plan into one so we can commit all files together
        let coalesce_partitions = Arc::new(CoalescePartitionsExec::new(write_plan));

        Ok(Arc::new(IcebergCommitExec::new(
            table,
            self.catalog.clone(),
            coalesce_partitions,
            self.schema.clone(),
        )))
    }
}

fn resolve_partitioning_enabled(
    key_kind: Option<bucketing::PartitionKeysKind>,
    scan_config: Option<&IcebergScanConfig>,
    table_properties: &HashMap<String, String>,
) -> DFResult<bool> {
    match key_kind {
        Some(bucketing::PartitionKeysKind::Identity) => resolve_partitioning_family(
            scan_config.map(|config| config.value_partitioning),
            table_properties,
            TableProperties::PROPERTY_DATAFUSION_VALUE_PARTITIONING_ENABLED,
            TableProperties::PROPERTY_DATAFUSION_VALUE_PARTITIONING_ENABLED_DEFAULT,
        ),
        Some(bucketing::PartitionKeysKind::Bucket) => resolve_partitioning_family(
            scan_config.map(|config| config.bucket_execution),
            table_properties,
            TableProperties::PROPERTY_DATAFUSION_BUCKET_EXECUTION_ENABLED,
            TableProperties::PROPERTY_DATAFUSION_BUCKET_EXECUTION_ENABLED_DEFAULT,
        ),
        None => Ok(false),
    }
}

fn resolve_partitioning_family(
    override_value: Option<PartitioningOverride>,
    table_properties: &HashMap<String, String>,
    property_name: &str,
    default_value: bool,
) -> DFResult<bool> {
    match override_value {
        Some(PartitioningOverride::Enabled) => Ok(true),
        Some(PartitioningOverride::Disabled) => Ok(false),
        Some(PartitioningOverride::Auto) | None => table_properties
            .get(property_name)
            .map(|value| parse_bool_table_property(property_name, value))
            .transpose()
            .map(|value| value.unwrap_or(default_value)),
    }
}

fn parse_bool_table_property(property_name: &str, value: &str) -> DFResult<bool> {
    value.parse::<bool>().map_err(|e| {
        DataFusionError::Plan(format!(
            "Invalid value for {property_name}, expected 'true' or 'false': {e}"
        ))
    })
}

/// Static table provider for read-only snapshot access.
///
/// This provider holds a cached table instance and does not refresh metadata or support
/// write operations. Use this for consistent analytical queries, time-travel scenarios,
/// or when you want to avoid catalog overhead.
///
/// For catalog-backed tables with write support and automatic refresh, use
/// [`IcebergTableProvider`] instead.
#[derive(Debug, Clone)]
pub struct IcebergStaticTableProvider {
    /// The static table instance (never refreshed)
    table: Table,
    /// Optional snapshot ID for this static view
    snapshot_id: Option<i64>,
    /// A reference-counted arrow `Schema`
    schema: ArrowSchemaRef,
}

impl IcebergStaticTableProvider {
    /// Creates a static provider from a table instance.
    ///
    /// Uses the table's current snapshot for all queries. Does not support write operations.
    pub async fn try_new_from_table(table: Table) -> Result<Self> {
        let schema = Arc::new(schema_to_arrow_schema(table.metadata().current_schema())?);
        Ok(IcebergStaticTableProvider {
            table,
            snapshot_id: None,
            schema,
        })
    }

    /// Creates a static provider for a specific table snapshot.
    ///
    /// Queries the specified snapshot for all operations. Useful for time-travel queries.
    /// Does not support write operations.
    pub async fn try_new_from_table_snapshot(table: Table, snapshot_id: i64) -> Result<Self> {
        let snapshot = table
            .metadata()
            .snapshot_by_id(snapshot_id)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!(
                        "snapshot id {snapshot_id} not found in table {}",
                        table.identifier().name()
                    ),
                )
            })?;
        let table_schema = snapshot.schema(table.metadata())?;
        let schema = Arc::new(schema_to_arrow_schema(&table_schema)?);
        Ok(IcebergStaticTableProvider {
            table,
            snapshot_id: Some(snapshot_id),
            schema,
        })
    }
}

#[async_trait]
impl TableProvider for IcebergStaticTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> ArrowSchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(IcebergTableScan::new(
            self.table.clone(),
            self.snapshot_id,
            self.schema.clone(),
            projection,
            filters,
            limit,
        )))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // Push down all filters, as a single source of truth, the scanner will drop the filters which couldn't be push down
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn insert_into(
        &self,
        _state: &dyn Session,
        _input: Arc<dyn ExecutionPlan>,
        _insert_op: InsertOp,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Err(to_datafusion_error(Error::new(
            ErrorKind::FeatureUnsupported,
            "Write operations are not supported on IcebergStaticTableProvider. \
             Use IcebergTableProvider with a catalog for write support."
                .to_string(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use datafusion::common::Column;
    use datafusion::common::config::ConfigOptions;
    use datafusion::physical_expr::expressions::Column as PhysicalColumn;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::physical_plan::Partitioning;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use iceberg::io::FileIO;
    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
    use iceberg::table::{StaticTable, Table};
    use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
    use tempfile::TempDir;

    use crate::{IcebergScanConfig, PartitioningOverride};

    #[test]
    fn test_iceberg_scan_config_can_be_set_through_config_options() {
        let mut config_options = ConfigOptions::default();
        config_options
            .extensions
            .insert(IcebergScanConfig::default());

        config_options
            .set("iceberg.value_partitioning", "enabled")
            .unwrap();
        config_options
            .set("iceberg.bucket_execution", "disabled")
            .unwrap();

        let scan_config = config_options
            .extensions
            .get::<IcebergScanConfig>()
            .unwrap();
        assert_eq!(
            scan_config.value_partitioning,
            PartitioningOverride::Enabled
        );
        assert_eq!(scan_config.bucket_execution, PartitioningOverride::Disabled);
    }

    use super::*;

    async fn get_test_table_from_metadata_file() -> Table {
        let metadata_file_name = "TableMetadataV2Valid.json";
        let metadata_file_path = format!(
            "{}/tests/test_data/{}",
            env!("CARGO_MANIFEST_DIR"),
            metadata_file_name
        );
        let file_io = FileIO::new_with_fs();
        let static_identifier = TableIdent::from_strs(["static_ns", "static_table"]).unwrap();
        let static_table =
            StaticTable::from_metadata_file(&metadata_file_path, static_identifier, file_io)
                .await
                .unwrap();
        static_table.into_table()
    }

    async fn get_test_catalog_and_table() -> (Arc<dyn Catalog>, NamespaceIdent, String, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let warehouse_path = temp_dir.path().to_str().unwrap().to_string();

        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
            )
            .await
            .unwrap();

        let namespace = NamespaceIdent::new("test_ns".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();

        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let table_creation = TableCreation::builder()
            .name("test_table".to_string())
            .location(format!("{warehouse_path}/test_table"))
            .schema(schema)
            .properties(HashMap::new())
            .build();

        catalog
            .create_table(&namespace, table_creation)
            .await
            .unwrap();

        (
            Arc::new(catalog),
            namespace,
            "test_table".to_string(),
            temp_dir,
        )
    }

    // Tests for IcebergStaticTableProvider

    #[tokio::test]
    async fn test_static_provider_from_table() {
        let table = get_test_table_from_metadata_file().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table.clone())
            .await
            .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("mytable", Arc::new(table_provider))
            .unwrap();
        let df = ctx.sql("SELECT * FROM mytable").await.unwrap();
        let df_schema = df.schema();
        let df_columns = df_schema.fields();
        assert_eq!(df_columns.len(), 3);
        let x_column = df_columns.first().unwrap();
        let column_data = format!(
            "{:?}:{:?}",
            x_column.name(),
            x_column.data_type().to_string()
        );
        assert_eq!(column_data, "\"x\":\"Int64\"");
        let has_column = df_schema.has_column(&Column::from_name("z"));
        assert!(has_column);
    }

    #[tokio::test]
    async fn test_static_provider_from_snapshot() {
        let table = get_test_table_from_metadata_file().await;
        let snapshot_id = table.metadata().snapshots().next().unwrap().snapshot_id();
        let table_provider =
            IcebergStaticTableProvider::try_new_from_table_snapshot(table.clone(), snapshot_id)
                .await
                .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("mytable", Arc::new(table_provider))
            .unwrap();
        let df = ctx.sql("SELECT * FROM mytable").await.unwrap();
        let df_schema = df.schema();
        let df_columns = df_schema.fields();
        assert_eq!(df_columns.len(), 3);
        let x_column = df_columns.first().unwrap();
        let column_data = format!(
            "{:?}:{:?}",
            x_column.name(),
            x_column.data_type().to_string()
        );
        assert_eq!(column_data, "\"x\":\"Int64\"");
        let has_column = df_schema.has_column(&Column::from_name("z"));
        assert!(has_column);
    }

    #[tokio::test]
    async fn test_static_provider_rejects_writes() {
        let table = get_test_table_from_metadata_file().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table.clone())
            .await
            .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("mytable", Arc::new(table_provider))
            .unwrap();

        // Attempt to insert into the static provider should fail
        let result = ctx.sql("INSERT INTO mytable VALUES (1, 2, 3)").await;

        // The error should occur during planning or execution
        // We expect an error indicating write operations are not supported
        assert!(
            result.is_err() || {
                let df = result.unwrap();
                df.collect().await.is_err()
            }
        );
    }

    #[tokio::test]
    async fn test_static_provider_scan() {
        let table = get_test_table_from_metadata_file().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table.clone())
            .await
            .unwrap();
        let ctx = SessionContext::new();
        ctx.register_table("mytable", Arc::new(table_provider))
            .unwrap();

        // Test that scan operations work correctly
        let df = ctx.sql("SELECT count(*) FROM mytable").await.unwrap();
        let physical_plan = df.create_physical_plan().await;
        assert!(physical_plan.is_ok());
    }

    // Tests for IcebergTableProvider

    #[tokio::test]
    async fn test_catalog_backed_provider_creation() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;

        // Test creating a catalog-backed provider
        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        // Verify the schema is loaded correctly
        let schema = provider.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "name");
    }

    #[tokio::test]
    async fn test_catalog_backed_provider_scan() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        ctx.register_table("test_table", Arc::new(provider))
            .unwrap();

        // Test that scan operations work correctly
        let df = ctx.sql("SELECT * FROM test_table").await.unwrap();

        // Verify the schema in the query result
        let df_schema = df.schema();
        assert_eq!(df_schema.fields().len(), 2);
        assert_eq!(df_schema.field(0).name(), "id");
        assert_eq!(df_schema.field(1).name(), "name");

        let physical_plan = df.create_physical_plan().await;
        assert!(physical_plan.is_ok());
    }

    #[tokio::test]
    async fn test_catalog_backed_provider_insert() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        ctx.register_table("test_table", Arc::new(provider))
            .unwrap();

        // Test that insert operations work correctly
        let result = ctx.sql("INSERT INTO test_table VALUES (1, 'test')").await;

        // Insert should succeed (or at least not fail during planning)
        assert!(result.is_ok());

        // Try to execute the insert plan
        let df = result.unwrap();
        let execution_result = df.collect().await;

        // The execution should succeed
        assert!(execution_result.is_ok());
    }

    #[tokio::test]
    async fn test_physical_input_schema_consistent_with_logical_input_schema() {
        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        ctx.register_table("test_table", Arc::new(provider))
            .unwrap();

        // Create a query plan
        let df = ctx.sql("SELECT id, name FROM test_table").await.unwrap();

        // Get logical schema before consuming df
        let logical_schema = df.schema().clone();

        // Get physical plan (this consumes df)
        let physical_plan = df.create_physical_plan().await.unwrap();
        let physical_schema = physical_plan.schema();

        // Verify that logical and physical schemas are consistent
        assert_eq!(
            logical_schema.fields().len(),
            physical_schema.fields().len()
        );

        for (logical_field, physical_field) in logical_schema
            .fields()
            .iter()
            .zip(physical_schema.fields().iter())
        {
            assert_eq!(logical_field.name(), physical_field.name());
            assert_eq!(logical_field.data_type(), physical_field.data_type());
        }
    }

    async fn get_partitioned_test_catalog_and_table(
        fanout_enabled: Option<bool>,
    ) -> (Arc<dyn Catalog>, NamespaceIdent, String, TempDir) {
        use iceberg::spec::{Transform, UnboundPartitionSpec};

        let temp_dir = TempDir::new().unwrap();
        let warehouse_path = temp_dir.path().to_str().unwrap().to_string();

        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse_path.clone())]),
            )
            .await
            .unwrap();

        let namespace = NamespaceIdent::new("test_ns".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();

        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(2, "category", Transform::Identity)
            .unwrap()
            .build();

        let mut properties = HashMap::new();
        if let Some(enabled) = fanout_enabled {
            properties.insert(
                iceberg::spec::TableProperties::PROPERTY_DATAFUSION_WRITE_FANOUT_ENABLED
                    .to_string(),
                enabled.to_string(),
            );
        }

        let table_creation = TableCreation::builder()
            .name("partitioned_table".to_string())
            .location(format!("{warehouse_path}/partitioned_table"))
            .schema(schema)
            .partition_spec(partition_spec)
            .properties(properties)
            .build();

        catalog
            .create_table(&namespace, table_creation)
            .await
            .unwrap();

        (
            Arc::new(catalog),
            namespace,
            "partitioned_table".to_string(),
            temp_dir,
        )
    }

    /// Helper to check if a plan contains a SortExec node
    fn plan_contains_sort(plan: &Arc<dyn ExecutionPlan>) -> bool {
        if plan.name() == "SortExec" {
            return true;
        }
        for child in plan.children() {
            if plan_contains_sort(child) {
                return true;
            }
        }
        false
    }

    #[tokio::test]
    async fn test_insert_plan_fanout_enabled_no_sort() {
        use datafusion::datasource::TableProvider;
        use datafusion::logical_expr::dml::InsertOp;
        use datafusion::physical_plan::empty::EmptyExec;

        // When fanout is enabled (default), no sort node should be added
        let (catalog, namespace, table_name, _temp_dir) =
            get_partitioned_test_catalog_and_table(Some(true)).await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        let input_schema = provider.schema();
        let input = Arc::new(EmptyExec::new(input_schema)) as Arc<dyn ExecutionPlan>;

        let state = ctx.state();
        let insert_plan = provider
            .insert_into(&state, input, InsertOp::Append)
            .await
            .unwrap();

        // With fanout enabled, there should be no SortExec in the plan
        assert!(
            !plan_contains_sort(&insert_plan),
            "Plan should NOT contain SortExec when fanout is enabled"
        );
    }

    #[tokio::test]
    async fn test_insert_plan_fanout_disabled_has_sort() {
        use datafusion::datasource::TableProvider;
        use datafusion::logical_expr::dml::InsertOp;
        use datafusion::physical_plan::empty::EmptyExec;

        // When fanout is disabled, a sort node should be added
        let (catalog, namespace, table_name, _temp_dir) =
            get_partitioned_test_catalog_and_table(Some(false)).await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        let input_schema = provider.schema();
        let input = Arc::new(EmptyExec::new(input_schema)) as Arc<dyn ExecutionPlan>;

        let state = ctx.state();
        let insert_plan = provider
            .insert_into(&state, input, InsertOp::Append)
            .await
            .unwrap();

        // With fanout disabled, there should be a SortExec in the plan
        assert!(
            plan_contains_sort(&insert_plan),
            "Plan should contain SortExec when fanout is disabled"
        );
    }

    #[tokio::test]
    async fn test_limit_pushdown_static_provider() {
        use datafusion::datasource::TableProvider;

        let table = get_test_table_from_metadata_file().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table.clone())
            .await
            .unwrap();

        let ctx = SessionContext::new();
        let state = ctx.state();

        // Test scan with limit
        let scan_plan = table_provider
            .scan(&state, None, &[], Some(10))
            .await
            .unwrap();

        // Verify that the scan plan is an IcebergTableScan
        let iceberg_scan = scan_plan
            .as_any()
            .downcast_ref::<IcebergTableScan>()
            .expect("Expected IcebergTableScan");

        // Verify the limit is set
        assert_eq!(
            iceberg_scan.limit(),
            Some(10),
            "Limit should be set to 10 in the scan plan"
        );
    }

    #[tokio::test]
    async fn test_limit_pushdown_catalog_backed_provider() {
        use datafusion::datasource::TableProvider;

        let (catalog, namespace, table_name, _temp_dir) = get_test_catalog_and_table().await;

        let provider =
            IcebergTableProvider::try_new(catalog.clone(), namespace.clone(), table_name.clone())
                .await
                .unwrap();

        let ctx = SessionContext::new();
        let state = ctx.state();

        // Test scan with limit
        let scan_plan = provider.scan(&state, None, &[], Some(5)).await.unwrap();

        // Verify that the scan plan is an IcebergTableScan
        let iceberg_scan = scan_plan
            .as_any()
            .downcast_ref::<IcebergTableScan>()
            .expect("Expected IcebergTableScan");

        // Verify the limit is set
        assert_eq!(
            iceberg_scan.limit(),
            Some(5),
            "Limit should be set to 5 in the scan plan"
        );
    }

    #[tokio::test]
    async fn test_no_limit_pushdown() {
        use datafusion::datasource::TableProvider;

        let table = get_test_table_from_metadata_file().await;
        let table_provider = IcebergStaticTableProvider::try_new_from_table(table.clone())
            .await
            .unwrap();

        let ctx = SessionContext::new();
        let state = ctx.state();

        // Test scan without limit
        let scan_plan = table_provider.scan(&state, None, &[], None).await.unwrap();

        // Verify that the scan plan is an IcebergTableScan
        let iceberg_scan = scan_plan
            .as_any()
            .downcast_ref::<IcebergTableScan>()
            .expect("Expected IcebergTableScan");

        // Verify the limit is None
        assert_eq!(
            iceberg_scan.limit(),
            None,
            "Limit should be None when not specified"
        );
    }

    // ── Bucketed scan tests ──────────────────────────────────────────────────

    async fn make_catalog_and_table_for_bucketing()
    -> (Arc<dyn Catalog>, NamespaceIdent, String, tempfile::TempDir) {
        use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
        use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
        use iceberg::{CatalogBuilder, TableCreation};

        let temp_dir = tempfile::TempDir::new().unwrap();
        let warehouse = temp_dir.path().to_str().unwrap().to_string();

        let catalog = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    std::collections::HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        warehouse.clone(),
                    )]),
                )
                .await
                .unwrap(),
        );

        let namespace = NamespaceIdent::new("ns".to_string());
        catalog
            .create_namespace(&namespace, std::collections::HashMap::new())
            .await
            .unwrap();

        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("t".to_string())
                    .location(format!("{warehouse}/t"))
                    .schema(schema)
                    .properties(std::collections::HashMap::new())
                    .build(),
            )
            .await
            .unwrap();

        (catalog, namespace, "t".to_string(), temp_dir)
    }

    /// Registers `n` synthetic data files in the table metadata via the iceberg
    /// transaction API. No actual parquet files are written, only the metadata
    /// entries that `plan_files()` reads are created.
    async fn append_fake_data_files(
        catalog: &Arc<dyn Catalog>,
        namespace: &NamespaceIdent,
        table_name: &str,
        n: usize,
    ) {
        use iceberg::spec::{DataContentType, DataFileBuilder, DataFileFormat};
        use iceberg::transaction::{ApplyTransactionAction, Transaction};

        let table = catalog
            .load_table(&TableIdent::new(namespace.clone(), table_name.to_string()))
            .await
            .unwrap();

        let data_files = (0..n)
            .map(|i| {
                DataFileBuilder::default()
                    .content(DataContentType::Data)
                    .file_path(format!(
                        "{}/data/fake_{i}.parquet",
                        table.metadata().location()
                    ))
                    .file_format(DataFileFormat::Parquet)
                    .file_size_in_bytes(128)
                    .record_count(1)
                    .partition_spec_id(table.metadata().default_partition_spec_id())
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(data_files);
        action
            .apply(tx)
            .unwrap()
            .commit(catalog.as_ref())
            .await
            .unwrap();
    }

    fn ctx_with_target_partitions(n: usize) -> SessionContext {
        ctx_with_target_partitions_and_scan_config(n, IcebergScanConfig::default())
    }

    fn ctx_with_target_partitions_and_scan_config(
        n: usize,
        scan_config: IcebergScanConfig,
    ) -> SessionContext {
        SessionContext::new_with_config(
            SessionConfig::new()
                .with_target_partitions(n)
                .with_option_extension(scan_config),
        )
    }

    /// An empty table must produce a single empty-bucket scan so that DataFusion
    /// can schedule the plan normally. execute(0) on an empty bucket simply
    /// returns an empty record-batch stream.
    #[tokio::test]
    async fn test_empty_table_single_empty_bucket() {
        let (catalog, namespace, table_name, _temp_dir) =
            make_catalog_and_table_for_bucketing().await;
        // no files appended
        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        let plan = provider
            .scan(&ctx_with_target_partitions(8).state(), None, &[], None)
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        assert_eq!(scan.buckets().len(), 1);
        assert_eq!(scan.buckets()[0].len(), 0);
        assert_eq!(scan.properties().partitioning.partition_count(), 1);
    }

    /// When the table has no identity-partition columns, every task takes the
    /// fallback (file_path) bucket path, so the declaration must drop to
    /// `UnknownPartitioning`. The bucket count should still equal
    /// min(target_partitions, num_files).
    #[tokio::test]
    async fn test_unpartitioned_falls_back_to_unknown() {
        use datafusion::physical_plan::Partitioning;

        let (catalog, namespace, table_name, _temp_dir) =
            make_catalog_and_table_for_bucketing().await;
        append_fake_data_files(&catalog, &namespace, &table_name, 5).await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        let plan = provider
            .scan(&ctx_with_target_partitions(3).state(), None, &[], None)
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        let total_files: usize = scan.buckets().iter().map(|b| b.len()).sum();
        assert_eq!(total_files, 5);
        assert_eq!(scan.buckets().len(), 3);
        assert!(matches!(
            scan.properties().partitioning,
            Partitioning::UnknownPartitioning(3)
        ));
    }

    /// Bucket count must be capped at the number of files: spinning up more
    /// DataFusion partitions than there are tasks would just leave empty
    /// streams, wasting scheduler slots.
    #[tokio::test]
    async fn test_bucket_count_capped_at_file_count() {
        let (catalog, namespace, table_name, _temp_dir) =
            make_catalog_and_table_for_bucketing().await;
        append_fake_data_files(&catalog, &namespace, &table_name, 2).await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        let plan = provider
            .scan(&ctx_with_target_partitions(16).state(), None, &[], None)
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        assert_eq!(scan.buckets().len(), 2);
    }

    /// target_partitions = 1 collapses every task into a single bucket, giving
    /// the same execution profile as a single-partition scan.
    #[tokio::test]
    async fn test_single_target_partition_single_bucket() {
        let (catalog, namespace, table_name, _temp_dir) =
            make_catalog_and_table_for_bucketing().await;
        append_fake_data_files(&catalog, &namespace, &table_name, 4).await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        let plan = provider
            .scan(&ctx_with_target_partitions(1).state(), None, &[], None)
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        assert_eq!(scan.buckets().len(), 1);
        assert_eq!(scan.buckets()[0].len(), 4);
    }

    async fn make_partitioned_catalog_and_table_for_bucketing()
    -> (Arc<dyn Catalog>, NamespaceIdent, String, tempfile::TempDir) {
        make_partitioned_catalog_and_table_for_bucketing_with_properties(HashMap::new()).await
    }

    async fn make_partitioned_catalog_and_table_for_bucketing_with_properties(
        properties: HashMap<String, String>,
    ) -> (Arc<dyn Catalog>, NamespaceIdent, String, tempfile::TempDir) {
        use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
        use iceberg::spec::{
            NestedField, PrimitiveType, Schema, Transform, Type, UnboundPartitionSpec,
        };
        use iceberg::{CatalogBuilder, TableCreation};

        let temp_dir = tempfile::TempDir::new().unwrap();
        let warehouse = temp_dir.path().to_str().unwrap().to_string();

        let catalog = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    std::collections::HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        warehouse.clone(),
                    )]),
                )
                .await
                .unwrap(),
        );

        let namespace = NamespaceIdent::new("ns".to_string());
        catalog
            .create_namespace(&namespace, std::collections::HashMap::new())
            .await
            .unwrap();

        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(2, "name_part", Transform::Identity)
            .unwrap()
            .build();

        catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("t".to_string())
                    .location(format!("{warehouse}/t"))
                    .schema(schema)
                    .partition_spec(partition_spec)
                    .properties(properties)
                    .build(),
            )
            .await
            .unwrap();

        (catalog, namespace, "t".to_string(), temp_dir)
    }

    /// Like [`append_fake_data_files`] but each file carries a partition tuple
    /// matching the table's identity-partition spec on `name`.
    async fn append_partitioned_fake_data_files(
        catalog: &Arc<dyn Catalog>,
        namespace: &NamespaceIdent,
        table_name: &str,
        partition_values: Vec<&str>,
    ) {
        use iceberg::spec::{DataContentType, DataFileBuilder, DataFileFormat, Literal, Struct};
        use iceberg::transaction::{ApplyTransactionAction, Transaction};

        let table = catalog
            .load_table(&TableIdent::new(namespace.clone(), table_name.to_string()))
            .await
            .unwrap();

        let data_files = partition_values
            .iter()
            .enumerate()
            .map(|(i, value)| {
                DataFileBuilder::default()
                    .content(DataContentType::Data)
                    .file_path(format!(
                        "{}/data/fake_{i}.parquet",
                        table.metadata().location()
                    ))
                    .file_format(DataFileFormat::Parquet)
                    .file_size_in_bytes(128)
                    .record_count(1)
                    .partition_spec_id(table.metadata().default_partition_spec_id())
                    .partition(Struct::from_iter(vec![Some(Literal::string(*value))]))
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(data_files);
        action
            .apply(tx)
            .unwrap()
            .commit(catalog.as_ref())
            .await
            .unwrap();
    }

    /// Identity partitioning is opt-in because value distribution can be skewed.
    #[tokio::test]
    async fn test_identity_partitioned_defaults_to_unknown() {
        let (catalog, namespace, table_name, _temp_dir) =
            make_partitioned_catalog_and_table_for_bucketing().await;
        append_partitioned_fake_data_files(
            &catalog,
            &namespace,
            &table_name,
            vec!["a", "b", "c", "a", "b", "c"],
        )
        .await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        let plan = provider
            .scan(&ctx_with_target_partitions(3).state(), None, &[], None)
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        let total_files: usize = scan.buckets().iter().map(|b| b.len()).sum();
        assert_eq!(total_files, 6);

        assert!(matches!(
            scan.properties().partitioning,
            Partitioning::UnknownPartitioning(3)
        ));
    }

    /// A table property can opt identity partitioning into hash declaration.
    #[tokio::test]
    async fn test_identity_partitioned_property_enabled_declares_hash() {
        let mut properties = HashMap::new();
        properties.insert(
            TableProperties::PROPERTY_DATAFUSION_VALUE_PARTITIONING_ENABLED.to_string(),
            "true".to_string(),
        );
        let (catalog, namespace, table_name, _temp_dir) =
            make_partitioned_catalog_and_table_for_bucketing_with_properties(properties).await;
        append_partitioned_fake_data_files(
            &catalog,
            &namespace,
            &table_name,
            vec!["a", "b", "c", "a", "b", "c"],
        )
        .await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        let plan = provider
            .scan(&ctx_with_target_partitions(3).state(), None, &[], None)
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        match &scan.properties().partitioning {
            Partitioning::Hash(exprs, n) => {
                assert_eq!(*n, 3);
                assert_eq!(exprs.len(), 1);
                let col = exprs[0]
                    .as_any()
                    .downcast_ref::<PhysicalColumn>()
                    .expect("expected Column expr");
                assert_eq!(col.name(), "name");
            }
            other => panic!("expected Partitioning::Hash, got {other:?}"),
        }
    }

    /// Session overrides take precedence over table properties.
    #[tokio::test]
    async fn test_identity_partitioned_session_disabled_overrides_property_enabled() {
        let mut properties = HashMap::new();
        properties.insert(
            TableProperties::PROPERTY_DATAFUSION_VALUE_PARTITIONING_ENABLED.to_string(),
            "true".to_string(),
        );
        let (catalog, namespace, table_name, _temp_dir) =
            make_partitioned_catalog_and_table_for_bucketing_with_properties(properties).await;
        append_partitioned_fake_data_files(&catalog, &namespace, &table_name, vec!["a", "b"]).await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        let ctx = ctx_with_target_partitions_and_scan_config(
            3,
            IcebergScanConfig {
                value_partitioning: PartitioningOverride::Disabled,
                ..Default::default()
            },
        );
        let plan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        assert!(matches!(
            scan.properties().partitioning,
            Partitioning::UnknownPartitioning(2)
        ));
    }

    #[tokio::test]
    async fn test_identity_partitioned_invalid_property_errors() {
        let mut properties = HashMap::new();
        properties.insert(
            TableProperties::PROPERTY_DATAFUSION_VALUE_PARTITIONING_ENABLED.to_string(),
            "sometimes".to_string(),
        );
        let (catalog, namespace, table_name, _temp_dir) =
            make_partitioned_catalog_and_table_for_bucketing_with_properties(properties).await;
        append_partitioned_fake_data_files(&catalog, &namespace, &table_name, vec!["a"]).await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        let err = provider
            .scan(&ctx_with_target_partitions(3).state(), None, &[], None)
            .await
            .unwrap_err();

        assert!(
            err.to_string()
                .contains(TableProperties::PROPERTY_DATAFUSION_VALUE_PARTITIONING_ENABLED)
        );
    }

    /// A projection that omits the partition source column drops
    /// `compute_identity_cols` to `None`, collapsing to `UnknownPartitioning`.
    #[tokio::test]
    async fn test_projection_without_partition_col_falls_back_to_unknown() {
        let (catalog, namespace, table_name, _temp_dir) =
            make_partitioned_catalog_and_table_for_bucketing().await;
        append_partitioned_fake_data_files(&catalog, &namespace, &table_name, vec!["a", "b"]).await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        // Project only "id" (idx 0), excluding the partition column "name" (idx 1).
        let projection = vec![0_usize];
        let plan = provider
            .scan(
                &ctx_with_target_partitions(3).state(),
                Some(&projection),
                &[],
                None,
            )
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        assert!(matches!(
            scan.properties().partitioning,
            Partitioning::UnknownPartitioning(_)
        ));
    }

    // ── Bucket-transform partitioning tests ─────────────────────────────────

    /// Build a table partitioned by `bucket[N](col)`. `transform` lets callers
    /// also build mixed-spec tables for negative tests.
    async fn make_bucket_partitioned_catalog_and_table_for_bucketing(
        n_buckets: u32,
    ) -> (Arc<dyn Catalog>, NamespaceIdent, String, tempfile::TempDir) {
        use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
        use iceberg::spec::{
            NestedField, PrimitiveType, Schema, Transform, Type, UnboundPartitionSpec,
        };
        use iceberg::{CatalogBuilder, TableCreation};

        let temp_dir = tempfile::TempDir::new().unwrap();
        let warehouse = temp_dir.path().to_str().unwrap().to_string();

        let catalog = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    std::collections::HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        warehouse.clone(),
                    )]),
                )
                .await
                .unwrap(),
        );

        let namespace = NamespaceIdent::new("ns".to_string());
        catalog
            .create_namespace(&namespace, std::collections::HashMap::new())
            .await
            .unwrap();

        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(2, "name_bucket", Transform::Bucket(n_buckets))
            .unwrap()
            .build();

        catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("t".to_string())
                    .location(format!("{warehouse}/t"))
                    .schema(schema)
                    .partition_spec(partition_spec)
                    .properties(std::collections::HashMap::new())
                    .build(),
            )
            .await
            .unwrap();

        (catalog, namespace, "t".to_string(), temp_dir)
    }

    /// Append synthetic data files whose partition slot carries a bucket
    /// *index* (an `i32`), matching what Iceberg writes for `Transform::Bucket`.
    async fn append_bucket_partitioned_fake_data_files(
        catalog: &Arc<dyn Catalog>,
        namespace: &NamespaceIdent,
        table_name: &str,
        bucket_indices: Vec<Option<i32>>,
    ) {
        use iceberg::spec::{DataContentType, DataFileBuilder, DataFileFormat, Literal, Struct};
        use iceberg::transaction::{ApplyTransactionAction, Transaction};

        let table = catalog
            .load_table(&TableIdent::new(namespace.clone(), table_name.to_string()))
            .await
            .unwrap();

        let data_files = bucket_indices
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                DataFileBuilder::default()
                    .content(DataContentType::Data)
                    .file_path(format!(
                        "{}/data/fake_{i}.parquet",
                        table.metadata().location()
                    ))
                    .file_format(DataFileFormat::Parquet)
                    .file_size_in_bytes(128)
                    .record_count(1)
                    .partition_spec_id(table.metadata().default_partition_spec_id())
                    .partition(Struct::from_iter(vec![idx.map(Literal::int)]))
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(data_files);
        action
            .apply(tx)
            .unwrap()
            .commit(catalog.as_ref())
            .await
            .unwrap();
    }

    /// Pure `Bucket[N]` spec with the source column in the projection: scan
    /// must declare `Partitioning::Hash` referencing that source column.
    #[tokio::test]
    async fn test_bucket_partitioned_declares_hash() {
        let (catalog, namespace, table_name, _temp_dir) =
            make_bucket_partitioned_catalog_and_table_for_bucketing(8).await;
        append_bucket_partitioned_fake_data_files(
            &catalog,
            &namespace,
            &table_name,
            vec![Some(0), Some(1), Some(2), Some(0), Some(7), Some(7)],
        )
        .await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        let plan = provider
            .scan(&ctx_with_target_partitions(4).state(), None, &[], None)
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        let total_files: usize = scan.buckets().iter().map(|b| b.len()).sum();
        assert_eq!(total_files, 6);

        match &scan.properties().partitioning {
            Partitioning::Hash(exprs, n) => {
                assert_eq!(*n, 4);
                assert_eq!(exprs.len(), 1);
                let col = exprs[0]
                    .as_any()
                    .downcast_ref::<PhysicalColumn>()
                    .expect("expected Column expr");
                assert_eq!(col.name(), "name");
            }
            other => panic!("expected Partitioning::Hash, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_bucket_partitioned_session_disabled_falls_back_to_unknown() {
        let (catalog, namespace, table_name, _temp_dir) =
            make_bucket_partitioned_catalog_and_table_for_bucketing(8).await;
        append_bucket_partitioned_fake_data_files(
            &catalog,
            &namespace,
            &table_name,
            vec![Some(0), Some(1), Some(2)],
        )
        .await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        let ctx = ctx_with_target_partitions_and_scan_config(
            4,
            IcebergScanConfig {
                bucket_execution: PartitioningOverride::Disabled,
                ..Default::default()
            },
        );
        let plan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        assert!(matches!(
            scan.properties().partitioning,
            Partitioning::UnknownPartitioning(3)
        ));
    }

    /// Single-column bucket spec where the projection excludes the *only*
    /// bucket source column: after filtering, `compute_bucket_cols` has zero
    /// surviving columns and returns `None`. Scan must declare
    /// `UnknownPartitioning`. This is the empty-intersection corner case of
    /// the partial-projection logic exercised by
    /// [`test_bucket_multi_column_partial_projection_declares_hash_on_subset`].
    #[tokio::test]
    async fn test_bucket_projection_drops_all_bucket_sources_falls_back_to_unknown() {
        let (catalog, namespace, table_name, _temp_dir) =
            make_bucket_partitioned_catalog_and_table_for_bucketing(4).await;
        append_bucket_partitioned_fake_data_files(
            &catalog,
            &namespace,
            &table_name,
            vec![Some(0), Some(1)],
        )
        .await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        // Project only "id" (idx 0); the bucket source "name" (idx 1) is excluded.
        let projection = vec![0_usize];
        let plan = provider
            .scan(
                &ctx_with_target_partitions(3).state(),
                Some(&projection),
                &[],
                None,
            )
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        assert!(matches!(
            scan.properties().partitioning,
            Partitioning::UnknownPartitioning(_)
        ));
    }

    /// A `None` partition slot makes `bucket_hash` return `None`, so the
    /// task takes the fallback path. Even a single such task forces the
    /// whole scan to drop to `UnknownPartitioning`.
    #[tokio::test]
    async fn test_bucket_with_null_partition_value_falls_back_to_unknown() {
        let (catalog, namespace, table_name, _temp_dir) =
            make_bucket_partitioned_catalog_and_table_for_bucketing(4).await;
        append_bucket_partitioned_fake_data_files(
            &catalog,
            &namespace,
            &table_name,
            vec![Some(0), None, Some(2)],
        )
        .await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        let plan = provider
            .scan(&ctx_with_target_partitions(3).state(), None, &[], None)
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        assert!(matches!(
            scan.properties().partitioning,
            Partitioning::UnknownPartitioning(_)
        ));
    }

    /// Mixed `Bucket[N] + Truncate(_)` spec: `compute_bucket_cols` rejects
    /// it because not every field is a bucket transform. Identity detection
    /// also yields zero columns. Final declaration is `UnknownPartitioning`.
    #[tokio::test]
    async fn test_mixed_bucket_and_other_transform_falls_back_to_unknown() {
        use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
        use iceberg::spec::{
            DataContentType, DataFileBuilder, DataFileFormat, Literal, NestedField, PrimitiveType,
            Schema, Struct, Transform, Type, UnboundPartitionSpec,
        };
        use iceberg::transaction::{ApplyTransactionAction, Transaction};
        use iceberg::{CatalogBuilder, TableCreation};

        let temp_dir = tempfile::TempDir::new().unwrap();
        let warehouse = temp_dir.path().to_str().unwrap().to_string();
        let catalog: Arc<dyn Catalog> = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    std::collections::HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        warehouse.clone(),
                    )]),
                )
                .await
                .unwrap(),
        );
        let namespace = NamespaceIdent::new("ns".to_string());
        catalog
            .create_namespace(&namespace, std::collections::HashMap::new())
            .await
            .unwrap();

        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(2, "name_bucket", Transform::Bucket(8))
            .unwrap()
            .add_partition_field(2, "name_trunc", Transform::Truncate(4))
            .unwrap()
            .build();

        catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("t".to_string())
                    .location(format!("{warehouse}/t"))
                    .schema(schema)
                    .partition_spec(partition_spec)
                    .properties(std::collections::HashMap::new())
                    .build(),
            )
            .await
            .unwrap();

        let table = catalog
            .load_table(&TableIdent::new(namespace.clone(), "t".to_string()))
            .await
            .unwrap();
        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(format!("{}/data/fake.parquet", table.metadata().location()))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(128)
            .record_count(1)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter(vec![
                Some(Literal::int(0)),
                Some(Literal::string("aaaa")),
            ]))
            .build()
            .unwrap();
        let tx = Transaction::new(&table);
        tx.fast_append()
            .add_data_files(vec![data_file])
            .apply(tx)
            .unwrap()
            .commit(catalog.as_ref())
            .await
            .unwrap();

        let provider = IcebergTableProvider::try_new(catalog, namespace, "t".to_string())
            .await
            .unwrap();
        let plan = provider
            .scan(&ctx_with_target_partitions(3).state(), None, &[], None)
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        assert!(matches!(
            scan.properties().partitioning,
            Partitioning::UnknownPartitioning(_)
        ));
    }

    /// When identity partitioning is enabled, mixed `Identity + Bucket` specs
    /// keep only identity columns as hash keys.
    #[tokio::test]
    async fn test_mixed_identity_and_bucket_keeps_identity_only_hash() {
        use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
        use iceberg::spec::{
            DataContentType, DataFileBuilder, DataFileFormat, Literal, NestedField, PrimitiveType,
            Schema, Struct, Transform, Type, UnboundPartitionSpec,
        };
        use iceberg::transaction::{ApplyTransactionAction, Transaction};
        use iceberg::{CatalogBuilder, TableCreation};

        let temp_dir = tempfile::TempDir::new().unwrap();
        let warehouse = temp_dir.path().to_str().unwrap().to_string();
        let catalog: Arc<dyn Catalog> = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    std::collections::HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        warehouse.clone(),
                    )]),
                )
                .await
                .unwrap(),
        );
        let namespace = NamespaceIdent::new("ns".to_string());
        catalog
            .create_namespace(&namespace, std::collections::HashMap::new())
            .await
            .unwrap();

        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "country", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::required(2, "customer_id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();
        let partition_spec = UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(1, "country", Transform::Identity)
            .unwrap()
            .add_partition_field(2, "customer_bucket", Transform::Bucket(10))
            .unwrap()
            .build();

        catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("t".to_string())
                    .location(format!("{warehouse}/t"))
                    .schema(schema)
                    .partition_spec(partition_spec)
                    .properties(std::collections::HashMap::new())
                    .build(),
            )
            .await
            .unwrap();

        let table = catalog
            .load_table(&TableIdent::new(namespace.clone(), "t".to_string()))
            .await
            .unwrap();
        let data_files = vec![
            (Some("us"), Some(1)),
            (Some("us"), Some(2)),
            (Some("fr"), Some(3)),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (country, bucket_idx))| {
            DataFileBuilder::default()
                .content(DataContentType::Data)
                .file_path(format!(
                    "{}/data/fake_{i}.parquet",
                    table.metadata().location()
                ))
                .file_format(DataFileFormat::Parquet)
                .file_size_in_bytes(128)
                .record_count(1)
                .partition_spec_id(table.metadata().default_partition_spec_id())
                .partition(Struct::from_iter(vec![
                    country.map(Literal::string),
                    bucket_idx.map(Literal::int),
                ]))
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>();
        let tx = Transaction::new(&table);
        tx.fast_append()
            .add_data_files(data_files)
            .apply(tx)
            .unwrap()
            .commit(catalog.as_ref())
            .await
            .unwrap();

        let provider = IcebergTableProvider::try_new(catalog, namespace, "t".to_string())
            .await
            .unwrap();
        let ctx = ctx_with_target_partitions_and_scan_config(
            3,
            IcebergScanConfig {
                value_partitioning: PartitioningOverride::Enabled,
                ..Default::default()
            },
        );
        let plan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        match &scan.properties().partitioning {
            Partitioning::Hash(exprs, n) => {
                assert_eq!(*n, 3);
                assert_eq!(
                    exprs.len(),
                    1,
                    "only the identity column should be retained"
                );
                let col = exprs[0]
                    .as_any()
                    .downcast_ref::<PhysicalColumn>()
                    .expect("expected Column expr");
                assert_eq!(col.name(), "country");
            }
            other => panic!("expected Partitioning::Hash, got {other:?}"),
        }
    }

    /// Pure `Bucket[N]` with `target_partitions == N`: tasks must land
    /// *deterministically* at `bucket_idx % n_partitions = bucket_idx`,
    /// so every scan partition holds exactly one file and none is empty.
    /// This is the regression test for the birthday-paradox empty
    /// partitions observed in the original investigation
    /// (`dd-notes/iceberg-bucket-rehash-investigation.md`).
    #[tokio::test]
    async fn test_bucket_n_eq_target_partitions_is_balanced() {
        let (catalog, namespace, table_name, _temp_dir) =
            make_bucket_partitioned_catalog_and_table_for_bucketing(8).await;
        // One file per bucket index 0..=7.
        append_bucket_partitioned_fake_data_files(
            &catalog,
            &namespace,
            &table_name,
            (0..8).map(Some).collect(),
        )
        .await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        let plan = provider
            .scan(&ctx_with_target_partitions(8).state(), None, &[], None)
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();
        let buckets = scan.buckets();

        assert_eq!(buckets.len(), 8);
        for (partition_idx, files) in buckets.iter().enumerate() {
            assert_eq!(
                files.len(),
                1,
                "partition {partition_idx} should hold exactly one file"
            );
            // Each file's stored partition slot must equal its scan-partition
            // index — the deterministic identity mapping when `N == n`.
            let slot = files[0]
                .partition
                .as_ref()
                .and_then(|s| s.fields().first().and_then(|f| f.clone()))
                .expect("partition slot must be present");
            let stored_idx = match slot {
                iceberg::spec::Literal::Primitive(iceberg::spec::PrimitiveLiteral::Int(v)) => v,
                other => panic!("unexpected slot literal: {other:?}"),
            };
            assert_eq!(
                stored_idx as usize, partition_idx,
                "bucket idx must map to its own scan partition"
            );
        }
    }

    /// Pure `Bucket[N]` with `target_partitions < N`: tasks land at
    /// `bucket_idx % target_partitions` deterministically.
    #[tokio::test]
    async fn test_bucket_n_gt_target_partitions_modulo_grouping() {
        let (catalog, namespace, table_name, _temp_dir) =
            make_bucket_partitioned_catalog_and_table_for_bucketing(8).await;
        append_bucket_partitioned_fake_data_files(
            &catalog,
            &namespace,
            &table_name,
            (0..8).map(Some).collect(),
        )
        .await;

        let provider = IcebergTableProvider::try_new(catalog, namespace, table_name)
            .await
            .unwrap();
        let plan = provider
            .scan(&ctx_with_target_partitions(3).state(), None, &[], None)
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();
        let buckets = scan.buckets();

        assert_eq!(buckets.len(), 3);
        // Expected grouping by `idx % 3`: {0,3,6} → p0, {1,4,7} → p1, {2,5} → p2.
        let expected: [&[i32]; 3] = [&[0, 3, 6], &[1, 4, 7], &[2, 5]];
        for (partition_idx, files) in buckets.iter().enumerate() {
            let mut got: Vec<i32> = files
                .iter()
                .map(|t| {
                    match t
                        .partition
                        .as_ref()
                        .and_then(|s| s.fields().first().and_then(|f| f.clone()))
                    {
                        Some(iceberg::spec::Literal::Primitive(
                            iceberg::spec::PrimitiveLiteral::Int(v),
                        )) => v,
                        other => panic!("unexpected slot: {other:?}"),
                    }
                })
                .collect();
            got.sort_unstable();
            assert_eq!(
                got,
                expected[partition_idx].to_vec(),
                "partition {partition_idx} grouping mismatch"
            );
        }
    }

    /// Multi-column pure-bucket spec `(bucket(2, name), bucket(4, id))`:
    /// even though every field is a bucket transform, multi-column specs
    /// are no longer hash-declarable. Scan must fall back to
    /// `Partitioning::UnknownPartitioning`.
    #[tokio::test]
    async fn test_multi_column_bucket_falls_back_to_unknown() {
        use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
        use iceberg::spec::{
            DataContentType, DataFileBuilder, DataFileFormat, Literal, NestedField, PrimitiveType,
            Schema, Struct, Transform, Type, UnboundPartitionSpec,
        };
        use iceberg::transaction::{ApplyTransactionAction, Transaction};
        use iceberg::{CatalogBuilder, TableCreation};

        let temp_dir = tempfile::TempDir::new().unwrap();
        let warehouse = temp_dir.path().to_str().unwrap().to_string();
        let catalog: Arc<dyn Catalog> = Arc::new(
            MemoryCatalogBuilder::default()
                .load(
                    "memory",
                    std::collections::HashMap::from([(
                        MEMORY_CATALOG_WAREHOUSE.to_string(),
                        warehouse.clone(),
                    )]),
                )
                .await
                .unwrap(),
        );
        let namespace = NamespaceIdent::new("ns".to_string());
        catalog
            .create_namespace(&namespace, std::collections::HashMap::new())
            .await
            .unwrap();

        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(2, "name_bucket", Transform::Bucket(2))
            .unwrap()
            .add_partition_field(1, "id_bucket", Transform::Bucket(4))
            .unwrap()
            .build();

        catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("t".to_string())
                    .location(format!("{warehouse}/t"))
                    .schema(schema)
                    .partition_spec(partition_spec)
                    .properties(std::collections::HashMap::new())
                    .build(),
            )
            .await
            .unwrap();

        let table = catalog
            .load_table(&TableIdent::new(namespace.clone(), "t".to_string()))
            .await
            .unwrap();

        let tuples: [(i32, i32); 4] = [(0, 0), (0, 1), (1, 0), (1, 3)];
        let data_files = tuples
            .iter()
            .enumerate()
            .map(|(i, &(name_idx, id_idx))| {
                DataFileBuilder::default()
                    .content(DataContentType::Data)
                    .file_path(format!(
                        "{}/data/fake_{i}.parquet",
                        table.metadata().location()
                    ))
                    .file_format(DataFileFormat::Parquet)
                    .file_size_in_bytes(128)
                    .record_count(1)
                    .partition_spec_id(table.metadata().default_partition_spec_id())
                    .partition(Struct::from_iter(vec![
                        Some(Literal::int(name_idx)),
                        Some(Literal::int(id_idx)),
                    ]))
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let tx = Transaction::new(&table);
        tx.fast_append()
            .add_data_files(data_files)
            .apply(tx)
            .unwrap()
            .commit(catalog.as_ref())
            .await
            .unwrap();

        let provider = IcebergTableProvider::try_new(catalog, namespace, "t".to_string())
            .await
            .unwrap();
        let plan = provider
            .scan(&ctx_with_target_partitions(4).state(), None, &[], None)
            .await
            .unwrap();
        let scan = plan.as_any().downcast_ref::<IcebergTableScan>().unwrap();

        match &scan.properties().partitioning {
            Partitioning::UnknownPartitioning(_) => {}
            other => panic!("expected Partitioning::UnknownPartitioning, got {other:?}"),
        }
    }
}
