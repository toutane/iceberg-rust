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

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, ExecutionPlan, Partitioning, PlanProperties};
use datafusion::prelude::Expr;
use futures::{Stream, TryStreamExt};
use iceberg::expr::Predicate;
use iceberg::scan::{FileScanTask, TableScan};
use iceberg::table::Table;

use super::expr_to_predicate::convert_filters_to_predicate;
use crate::table::PartitionKeysKind;
use crate::to_datafusion_error;

/// Iceberg [`Table`] scan as a DataFusion [`ExecutionPlan`].
///
/// Has three construction modes: [`new`][Self::new] (lazy, single-partition),
/// [`new_with_tasks`][Self::new_with_tasks] (eager, over pre-planned
/// [`FileScanTask`] buckets), and
/// [`new_with_tasks_from_predicate`][Self::new_with_tasks_from_predicate]
/// (eager, from a [`Predicate`]) — the last lets a node be rebuilt from its
/// getters, e.g. by a distributed-plan codec.
///
/// Note: in eager mode the underlying `TableScan` is rebuilt on every
/// `execute(partition)` call. The per-build cost is bounded (no I/O) and
/// keeps the plan free of `Arc`-shared evaluator caches that are awkward to
/// serialize across workers.
#[derive(Debug)]
pub struct IcebergTableScan {
    /// A table in the catalog.
    table: Table,
    /// Snapshot of the table to scan.
    snapshot_id: Option<i64>,
    /// Cached plan properties used by query optimization.
    plan_properties: Arc<PlanProperties>,
    /// Projection column names, None means all columns.
    projection: Option<Vec<String>>,
    /// Full schema before projection. Kept verbatim, not re-derived: the
    /// provider caches it while reloading the table, so it can diverge from the
    /// table's current metadata.
    table_schema: ArrowSchemaRef,
    /// Projection indices as received by `scan`; `projection` keeps only names.
    projection_indices: Option<Vec<usize>>,
    /// Filters to apply to the table scan.
    predicates: Option<Predicate>,
    /// Pre-planned file scan tasks per partition (eager mode), or `None` (lazy mode).
    buckets: Option<Vec<Vec<FileScanTask>>>,
    /// `None` when partitioning is `UnknownPartitioning`.
    partition_keys_kind: Option<PartitionKeysKind>,
    /// Optional limit on the number of rows to return.
    limit: Option<usize>,
}

impl IcebergTableScan {
    /// Creates a lazy single-partition scan that plans and reads all tasks
    /// inside `execute(0)`. Used by
    /// [`IcebergStaticTableProvider`][crate::table::IcebergStaticTableProvider].
    pub fn new(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Self {
        Self::new_inner(
            table,
            snapshot_id,
            schema,
            projection,
            convert_filters_to_predicate(filters),
            limit,
            Partitioning::UnknownPartitioning(1),
            None,
        )
    }

    /// Creates an eager multi-partition scan over pre-planned task buckets.
    /// Partition `i` streams `buckets[i]`. The caller is responsible for
    /// ensuring `partitioning` matches the bucketing. Used by
    /// [`IcebergTableProvider`][crate::table::IcebergTableProvider].
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_tasks(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
        buckets: Vec<Vec<FileScanTask>>,
        partitioning: Partitioning,
    ) -> Self {
        Self::new_inner(
            table,
            snapshot_id,
            schema,
            projection,
            convert_filters_to_predicate(filters),
            limit,
            partitioning,
            Some(buckets),
        )
    }

    /// Eager variant taking a [`Predicate`] instead of [`Expr`] filters, so a
    /// node can be rebuilt from its getters. The predicate is unbound; the scan
    /// builder binds it at `execute` time.
    // Arity mirrors `new_with_tasks`; an args struct is deferred.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_tasks_from_predicate(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        projection: Option<&Vec<usize>>,
        predicate: Option<Predicate>,
        limit: Option<usize>,
        buckets: Vec<Vec<FileScanTask>>,
        partitioning: Partitioning,
    ) -> Self {
        Self::new_inner(
            table,
            snapshot_id,
            schema,
            projection,
            predicate,
            limit,
            partitioning,
            Some(buckets),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        projection: Option<&Vec<usize>>,
        predicate: Option<Predicate>,
        limit: Option<usize>,
        partitioning: Partitioning,
        buckets: Option<Vec<Vec<FileScanTask>>>,
    ) -> Self {
        let output_schema = match projection {
            None => schema.clone(),
            Some(projection) => Arc::new(schema.project(projection).unwrap()),
        };
        let plan_properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(output_schema),
            partitioning,
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        let table_schema = Arc::clone(&schema);
        let projection_indices = projection.cloned();
        let projection = get_column_names(schema, projection);

        Self {
            table,
            snapshot_id,
            plan_properties,
            projection,
            table_schema,
            projection_indices,
            predicates: predicate,
            buckets,
            partition_keys_kind: None,
            limit,
        }
    }

    pub(crate) fn with_partition_keys_kind(
        mut self,
        partition_keys_kind: Option<PartitionKeysKind>,
    ) -> Self {
        self.partition_keys_kind = partition_keys_kind;
        self
    }

    pub fn table(&self) -> &Table {
        &self.table
    }

    pub fn table_schema(&self) -> &ArrowSchemaRef {
        &self.table_schema
    }

    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    pub fn projection(&self) -> Option<&[String]> {
        self.projection.as_deref()
    }

    pub fn projection_indices(&self) -> Option<&[usize]> {
        self.projection_indices.as_deref()
    }

    pub fn predicates(&self) -> Option<&Predicate> {
        self.predicates.as_ref()
    }

    /// Returns the pre-planned file task buckets, or an empty slice in lazy mode.
    pub fn buckets(&self) -> &[Vec<FileScanTask>] {
        self.buckets.as_deref().unwrap_or(&[])
    }

    /// Returns the transform family behind the `Partitioning::Hash` declaration,
    /// or `None` when the scan declares `UnknownPartitioning`.
    pub fn partition_keys_kind(&self) -> Option<PartitionKeysKind> {
        self.partition_keys_kind
    }

    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    fn total_file_count(&self) -> usize {
        self.buckets().iter().map(|b| b.len()).sum()
    }
}

impl ExecutionPlan for IcebergTableScan {
    fn name(&self) -> &str {
        "IcebergTableScan"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan + 'static>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Internal(format!(
                "{} is a leaf node and expects no children, but {} were provided",
                self.name(),
                children.len()
            )));
        }
        Ok(self)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.plan_properties
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let bucket = match &self.buckets {
            Some(buckets) => Some(buckets.get(partition).cloned().ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "{}: partition index {partition} is out of bounds (total buckets: {})",
                    self.name(),
                    buckets.len()
                ))
            })?),
            None => None,
        };

        let fut = build_record_batch_stream(
            self.table.clone(),
            self.snapshot_id,
            self.projection.clone(),
            self.predicates.clone(),
            bucket,
        );
        let stream = Box::pin(futures::stream::once(fut).try_flatten())
            as Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>>;

        let limited_stream = match self.limit {
            Some(limit) => apply_limit(stream, limit),
            None => stream,
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            limited_stream,
        )))
    }
}

impl DisplayAs for IcebergTableScan {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let projection = self
            .projection
            .as_deref()
            .map_or(String::new(), |v| v.join(","));
        let predicate = self
            .predicates
            .as_ref()
            .map_or(String::new(), |p| p.to_string());

        write!(
            f,
            "{} projection:[{projection}] predicate:[{predicate}]",
            self.name()
        )?;
        if let Some(buckets) = &self.buckets {
            let file_count = self.total_file_count();
            let bucket_count = buckets.len();
            write!(f, " buckets:[{bucket_count}] file_count:[{file_count}]")?;
        }
        if let Some(limit) = self.limit {
            write!(f, " limit:[{limit}]")?;
        }
        Ok(())
    }
}

fn build_table_scan(
    table: Table,
    snapshot_id: Option<i64>,
    column_names: Option<Vec<String>>,
    predicates: Option<Predicate>,
) -> DFResult<TableScan> {
    let scan_builder = match snapshot_id {
        Some(id) => table.scan().snapshot_id(id),
        None => table.scan(),
    };
    let mut scan_builder = match column_names {
        Some(names) => scan_builder.select(names),
        None => scan_builder.select_all(),
    };
    if let Some(pred) = predicates {
        scan_builder = scan_builder.with_filter(pred);
    }
    scan_builder.build().map_err(to_datafusion_error)
}

/// Builds the `RecordBatch` stream for a single partition. When `bucket` is
/// `Some`, streams the pre-planned tasks via `to_arrow_from_tasks`; when
/// `None`, plans and reads the full scan via `to_arrow`.
async fn build_record_batch_stream(
    table: Table,
    snapshot_id: Option<i64>,
    column_names: Option<Vec<String>>,
    predicates: Option<Predicate>,
    bucket: Option<Vec<FileScanTask>>,
) -> DFResult<Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>>> {
    let table_scan = build_table_scan(table, snapshot_id, column_names, predicates)?;
    let stream: Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>> = match bucket {
        Some(bucket) => {
            let task_stream = Box::pin(futures::stream::iter(
                bucket.into_iter().map(Ok::<_, iceberg::Error>),
            ));
            Box::pin(
                table_scan
                    .to_arrow_from_tasks(task_stream)
                    .map_err(to_datafusion_error)?
                    .map_err(to_datafusion_error),
            )
        }
        None => Box::pin(
            table_scan
                .to_arrow()
                .await
                .map_err(to_datafusion_error)?
                .map_err(to_datafusion_error),
        ),
    };
    Ok(stream)
}

/// Truncates a stream of `RecordBatch` to at most `limit` rows.
fn apply_limit(
    stream: Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>>,
    limit: usize,
) -> Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>> {
    let mut remaining = limit;
    Box::pin(stream.try_filter_map(move |batch| {
        futures::future::ready(if remaining == 0 {
            Ok(None)
        } else if batch.num_rows() <= remaining {
            remaining -= batch.num_rows();
            Ok(Some(batch))
        } else {
            let limited_batch = batch.slice(0, remaining);
            remaining = 0;
            Ok(Some(limited_batch))
        })
    }))
}

pub(super) fn get_column_names(
    schema: ArrowSchemaRef,
    projection: Option<&Vec<usize>>,
) -> Option<Vec<String>> {
    projection.map(|v| {
        v.iter()
            .map(|p| schema.field(*p).name().clone())
            .collect::<Vec<String>>()
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::physical_plan::ExecutionPlan;
    use iceberg::TableIdent;
    use iceberg::expr::Reference;
    use iceberg::io::FileIO;
    use iceberg::spec::Datum;
    use iceberg::table::{StaticTable, Table};

    use super::*;

    async fn get_test_table_from_metadata_file() -> Table {
        let metadata_file_name = "TableMetadataV2Valid.json";
        let metadata_file_path = format!(
            "{}/tests/test_data/{}",
            env!("CARGO_MANIFEST_DIR"),
            metadata_file_name
        );
        let ident = TableIdent::from_strs(["ns", "scan_table"]).unwrap();
        StaticTable::from_metadata_file(&metadata_file_path, ident, FileIO::new_with_fs())
            .await
            .unwrap()
            .into_table()
    }

    fn create_test_arrow_schema() -> ArrowSchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new("x", DataType::Int64, false),
            Field::new("y", DataType::Int64, false),
            Field::new("z", DataType::Int64, false),
        ]))
    }

    #[tokio::test]
    async fn test_predicate_constructor_exposes_rebuild_inputs() {
        let schema = create_test_arrow_schema();
        let projection = vec![0usize, 2];
        let predicate = Reference::new("x").greater_than(Datum::long(5));
        let scan = IcebergTableScan::new_with_tasks_from_predicate(
            get_test_table_from_metadata_file().await,
            None,
            schema.clone(),
            Some(&projection),
            Some(predicate.clone()),
            Some(100),
            vec![vec![], vec![]],
            Partitioning::UnknownPartitioning(2),
        );

        assert_eq!(scan.predicates(), Some(&predicate));
        assert_eq!(scan.table_schema().fields(), schema.fields());
        assert_eq!(scan.table_schema().fields().len(), 3);
        assert_eq!(scan.schema().fields().len(), 2);
        assert_ne!(scan.schema().fields(), scan.table_schema().fields());

        assert_eq!(scan.projection_indices(), Some(projection.as_slice()));
        let expected_projection = vec!["x".to_string(), "z".to_string()];
        assert_eq!(scan.projection(), Some(expected_projection.as_slice()));
        assert!(matches!(
            scan.properties().partitioning,
            Partitioning::UnknownPartitioning(2)
        ));
    }

    #[tokio::test]
    async fn test_no_projection_keeps_full_schema() {
        let schema = create_test_arrow_schema();
        let scan = IcebergTableScan::new_with_tasks_from_predicate(
            get_test_table_from_metadata_file().await,
            None,
            schema.clone(),
            None,
            None,
            None,
            vec![vec![]],
            Partitioning::UnknownPartitioning(1),
        );

        assert_eq!(scan.projection_indices(), None);
        assert_eq!(scan.projection(), None);
        assert_eq!(scan.predicates(), None);
        assert_eq!(scan.schema().fields(), scan.table_schema().fields());
        assert_eq!(scan.table_schema().fields(), schema.fields());
    }
}
