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

use std::sync::Arc;

use datafusion::arrow::datatypes::{
    DataType, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef, TimeUnit,
};
use datafusion::common::hash_utils::create_hashes;
use datafusion::error::Result as DFResult;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::Partitioning;
use datafusion::physical_plan::repartition::REPARTITION_RANDOM_STATE;
use datafusion::prelude::Expr;
use futures::TryStreamExt;
use iceberg::arrow::PrimitiveLiteralArrayBuilder;
use iceberg::expr::Predicate;
use iceberg::scan::{FileScanTask, TableScan};
use iceberg::spec::{Literal, Transform};
use iceberg::table::Table;

use super::expr_to_predicate::convert_filters_to_predicate;
use crate::to_datafusion_error;

#[derive(Debug, Clone)]
pub struct IcebergScanConfig {
    /// Snapshot of the table to scan.
    snapshot_id: Option<i64>,
    /// Output schema after projection.
    output_schema: ArrowSchemaRef,
    /// Projection column names, None means all columns.
    column_names: Option<Vec<String>>,
    /// Filters to apply to the table scan.
    predicates: Option<Predicate>,
}

#[derive(Debug)]
pub struct PlannedFileTaskGroups {
    pub groups: Vec<Vec<FileScanTask>>,
    pub partitioning: Partitioning,
}

/// Identity-partitioned column that is also present in the output projection
/// and whose Arrow type can be reconstructed from a `Literal` for hashing.
struct IdentityHashColumn {
    partition_spec_id: i32,
    name: String,
    /// Position of this column in the *output* schema (after projection).
    output_idx: usize,
    /// Position of this column inside the partition spec's `fields()` slice,
    /// matching the slot order of `FileScanTask::partition`.
    spec_field_idx: usize,
    output_dtype: DataType,
}

impl IcebergScanConfig {
    pub fn new(
        schema: ArrowSchemaRef,
        snapshot_id: Option<i64>,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
    ) -> Self {
        let output_schema = match projection {
            None => schema.clone(),
            Some(projection) => Arc::new(schema.project(projection).unwrap()),
        };

        Self {
            snapshot_id,
            output_schema,
            column_names: get_column_names(schema, projection),
            predicates: convert_filters_to_predicate(filters),
        }
    }

    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    pub fn output_schema(&self) -> ArrowSchemaRef {
        self.output_schema.clone()
    }

    pub fn column_names(&self) -> Option<&[String]> {
        self.column_names.as_deref()
    }

    pub fn predicates(&self) -> Option<&Predicate> {
        self.predicates.as_ref()
    }
}

pub(crate) async fn plan_file_task_groups(
    table: &Table,
    scan_config: &IcebergScanConfig,
    target_partitions: usize,
) -> DFResult<PlannedFileTaskGroups> {
    // Do not cache planned FileScanTasks in the provider in v1. They are query-specific
    // because projection, predicate binding, snapshot schema, and delete planning can differ
    // between scans. Catalog-backed providers also need fresh metadata on each scan.
    // TODO: Revisit provider-level caching for static tables with a precise cache key.
    let tasks: Vec<FileScanTask> = build_table_scan(table, scan_config)?
        .plan_files()
        .await
        .map_err(to_datafusion_error)?
        .try_collect::<Vec<_>>()
        .await
        .map_err(to_datafusion_error)?;

    let output_schema = scan_config.output_schema();
    Ok(plan_task_groups_from_tasks(
        table,
        output_schema.as_ref(),
        tasks,
        target_partitions,
    ))
}

fn plan_task_groups_from_tasks(
    table: &Table,
    output_schema: &ArrowSchema,
    tasks: Vec<FileScanTask>,
    target_partitions: usize,
) -> PlannedFileTaskGroups {
    if !tasks.is_empty()
        && let Some(identity_cols) = find_identity_hash_columns(table, output_schema)
    {
        let partition_count = target_partitions.min(tasks.len()).max(1);
        // Hash on a borrow first so that, on success, the tasks can be moved
        // (not cloned) into their target buckets.
        if let Some(hashes) = identity_hashes_for_tasks(&tasks, &identity_cols) {
            let mut groups: Vec<Vec<FileScanTask>> = vec![Vec::new(); partition_count];
            for (task, hash) in tasks.into_iter().zip(hashes) {
                groups[(hash % partition_count as u64) as usize].push(task);
            }
            let hash_exprs = identity_cols
                .iter()
                .map(|col| {
                    Arc::new(Column::new(&col.name, col.output_idx)) as Arc<dyn PhysicalExpr>
                })
                .collect();
            return PlannedFileTaskGroups {
                groups,
                partitioning: Partitioning::Hash(hash_exprs, partition_count),
            };
        }
    }

    let groups = group_file_scan_tasks_by_size(tasks, target_partitions);
    let partition_count = groups.len();
    PlannedFileTaskGroups {
        groups,
        partitioning: Partitioning::UnknownPartitioning(partition_count),
    }
}

/// Inspect the table's default partition spec and return the list of identity
/// columns that can support a [`Partitioning::Hash`] declaration. Returns
/// `None` if any condition is violated:
///   - the spec has no identity-transform field at all
///   - the source column for an identity field is not in the output projection
///   - the source column's Arrow type is not currently supported by
///     the identity hash materialization path
///   - the table has spec evolution (>1 historical specs), since older files
///     may carry a partition tuple that does not align with the default spec
///
/// Returning `None` forces the scan to declare `UnknownPartitioning` even if
/// bucketing succeeds.
fn find_identity_hash_columns(
    table: &Table,
    output_schema: &ArrowSchema,
) -> Option<Vec<IdentityHashColumn>> {
    let metadata = table.metadata();
    // iceberg-java is less conservative here: it intersects the identity fields
    // present in every spec (`Partitioning.groupingKeyType` /
    // `commonActiveFieldIds`) and still reports a grouping key on the columns
    // that are identity-partitioned across all of them. We deliberately bail
    // out on any spec evolution instead, because the bucketing path aligns each
    // task's partition slot to the *default* spec and `FileScanTask` does not
    // yet carry its own spec id to disambiguate. Tracked as a follow-up in
    // <https://github.com/apache/iceberg-rust/issues/2658>.
    if metadata.partition_specs_iter().len() != 1 {
        return None;
    }

    // Be conservative under schema evolution: the scan output schema can come
    // from a historical snapshot or from a provider-cached pre-evolution schema,
    // while the lookup below uses names from `metadata.current_schema()`. Under
    // rename/name reuse this can advertise hash partitioning for the wrong
    // output column. TODO: allow schema evolution here once identity columns
    // are matched by Iceberg field id metadata instead of current field name.
    if metadata.schemas_iter().len() != 1 {
        return None;
    }

    let table_schema = metadata.current_schema();
    let mut columns = Vec::new();
    for (spec_field_idx, partition_field) in metadata
        .default_partition_spec()
        .fields()
        .iter()
        .enumerate()
    {
        if partition_field.transform != Transform::Identity {
            continue;
        }

        let source_field = table_schema.field_by_id(partition_field.source_id)?;
        let source_path = table_schema.name_by_field_id(partition_field.source_id)?;
        if source_path.contains('.') {
            // TODO: Support hash partitioning for nested identity fields by
            // advertising partitioning on the exact nested output expression.
            // Matching by leaf name is unsafe when a top-level column shares it.
            return None;
        }

        let output_idx = output_schema.index_of(source_field.name.as_str()).ok()?;
        let output_field = output_schema.field(output_idx);
        if !is_supported_identity_hash_dtype(output_field.data_type()) {
            return None;
        }

        columns.push(IdentityHashColumn {
            partition_spec_id: metadata.default_partition_spec().spec_id(),
            name: output_field.name().clone(),
            output_idx,
            spec_field_idx,
            output_dtype: output_field.data_type().clone(),
        });
    }

    if columns.is_empty() {
        return None;
    }

    Some(columns)
}

fn is_supported_identity_hash_dtype(data_type: &DataType) -> bool {
    // Correctness here relies on the stored partition literal hashing identically
    // to the data column DataFusion sees at runtime. DataFusion hashes the
    // physical value (integer/native bits), so for floats the hash is over the
    // raw byte pattern: `+0.0`/`-0.0` and distinct NaN encodings hash differently.
    // This holds for a correctly written identity partition (the literal equals
    // every row's value bit-for-bit), but float identity keys are unusual; revisit
    // if a writer is found to normalize partition floats (e.g. `-0.0` -> `0.0`)
    // without rewriting the data, as that would split a key across partitions.
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Date32
            | DataType::Time64(TimeUnit::Microsecond)
            | DataType::Timestamp(TimeUnit::Microsecond, _)
            | DataType::Timestamp(TimeUnit::Nanosecond, _)
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::Decimal128(_, _)
            | DataType::FixedSizeBinary(_)
    )
}

/// Hash all identity-partition values using [`REPARTITION_RANDOM_STATE`] so the
/// bucket assignment matches DataFusion's hash-repartition convention.
fn identity_hashes_for_tasks(
    tasks: &[FileScanTask],
    identity_cols: &[IdentityHashColumn],
) -> Option<Vec<u64>> {
    if identity_cols.is_empty() {
        return None;
    }

    let mut builders = identity_cols
        .iter()
        .map(|col| PrimitiveLiteralArrayBuilder::try_new(&col.output_dtype, tasks.len()))
        .collect::<iceberg::Result<Vec<_>>>()
        .ok()?;
    let partition_spec_id = identity_cols.first()?.partition_spec_id;

    for task in tasks {
        if let Some(partition_spec) = task.partition_spec.as_ref()
            && partition_spec.spec_id() != partition_spec_id
        {
            return None;
        }

        let partition = task.partition.as_ref()?;
        for (builder, identity_col) in builders.iter_mut().zip(identity_cols) {
            let Some(Literal::Primitive(primitive)) = partition
                .fields()
                .get(identity_col.spec_field_idx)?
                .as_ref()
            else {
                return None;
            };
            if !builder.append_or_null(Some(primitive)).ok()? {
                return None;
            }
        }
    }

    let arrays = builders
        .into_iter()
        .map(PrimitiveLiteralArrayBuilder::finish)
        .collect::<iceberg::Result<Vec<_>>>()
        .ok()?;
    let mut hashes = vec![0; tasks.len()];
    create_hashes(
        &arrays,
        REPARTITION_RANDOM_STATE.random_state(),
        &mut hashes,
    )
    .ok()?;
    Some(hashes)
}

fn get_column_names(
    schema: ArrowSchemaRef,
    projection: Option<&Vec<usize>>,
) -> Option<Vec<String>> {
    projection.map(|v| {
        v.iter()
            .map(|p| schema.field(*p).name().clone())
            .collect::<Vec<String>>()
    })
}

/// Groups file scan tasks into `target_partitions` groups, greedily balancing
/// total file byte size across groups (longest-processing-time heuristic): tasks
/// are visited largest-first and each is appended to the currently lightest
/// group. Ties are broken by original index so the output is deterministic
/// regardless of input ordering. `target_partitions` is clamped to a minimum of
/// 1; empty groups are dropped.
///
/// Each task contributes [`task_weight`] (`max(file_size_in_bytes, 1)`): the
/// 1-byte floor keeps the distribution round-robin-like when sizes are equal or
/// unavailable (all-zero), instead of collapsing every task into the first group.
fn group_file_scan_tasks_by_size(
    tasks: Vec<FileScanTask>,
    target_partitions: usize,
) -> Vec<Vec<FileScanTask>> {
    if tasks.is_empty() {
        return vec![vec![]];
    }

    let target_partitions = target_partitions.max(1).min(tasks.len());

    let mut indexed: Vec<(usize, FileScanTask)> = tasks.into_iter().enumerate().collect();
    indexed.sort_by(|(ia, a), (ib, b)| task_weight(b).cmp(&task_weight(a)).then(ia.cmp(ib)));

    let mut groups: Vec<Vec<FileScanTask>> = vec![Vec::new(); target_partitions];
    let mut group_weight = vec![0u64; target_partitions];
    for (_, task) in indexed {
        // Pick the lightest group; `min_by_key` returns the first minimum on
        // ties, i.e. the lowest group index.
        let target = (0..target_partitions)
            .min_by_key(|&i| group_weight[i])
            .expect("target_partitions is at least 1");
        group_weight[target] += task_weight(&task);
        groups[target].push(task);
    }

    groups
}

/// Byte weight of a task, floored at 1 so equal or zero sizes still spread out
/// across groups instead of piling into the first one.
fn task_weight(task: &FileScanTask) -> u64 {
    task.file_size_in_bytes.max(1)
}

pub(crate) fn build_table_scan(
    table: &Table,
    scan_config: &IcebergScanConfig,
) -> DFResult<TableScan> {
    let builder = match scan_config.snapshot_id {
        Some(id) => table.scan().snapshot_id(id),
        None => table.scan(),
    };
    let mut builder = match scan_config.column_names.clone() {
        Some(names) => builder.select(names),
        None => builder.select_all(),
    };
    if let Some(pred) = scan_config.predicates.clone() {
        builder = builder.with_filter(pred);
    }
    builder.build().map_err(to_datafusion_error)
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{DataFileFormat, NestedField, PrimitiveType, Schema, Type};

    use super::*;

    /// Minimal `FileScanTask` carrying an explicit `file_size_in_bytes`.
    fn sized_task(path: &str, size: u64) -> FileScanTask {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "v", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();
        FileScanTask::builder()
            .with_file_size_in_bytes(size)
            .with_start(0)
            .with_length(size)
            .with_record_count(Some(1))
            .with_data_file_path(path.to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(Arc::new(schema))
            .with_project_field_ids(vec![1])
            .with_case_sensitive(true)
            .build()
    }

    fn task_paths(tasks: &[FileScanTask]) -> Vec<&str> {
        tasks.iter().map(|task| task.data_file_path()).collect()
    }

    fn group_bytes(groups: &[Vec<FileScanTask>]) -> Vec<u64> {
        groups
            .iter()
            .map(|group| group.iter().map(|task| task.file_size_in_bytes).sum())
            .collect()
    }

    #[test]
    fn test_size_based_grouping_balances_by_bytes() {
        let tasks = vec![
            sized_task("/a.parquet", 100),
            sized_task("/b.parquet", 90),
            sized_task("/c.parquet", 20),
            sized_task("/d.parquet", 10),
        ];

        let groups = group_file_scan_tasks_by_size(tasks, 2);

        // LPT over the descending sort [100, 90, 20, 10]:
        //   100 -> g0, 90 -> g1, 20 -> g1 (90 < 100), 10 -> g0 (100 < 110).
        assert_eq!(groups.len(), 2);
        assert_eq!(task_paths(&groups[0]), vec!["/a.parquet", "/d.parquet"]);
        assert_eq!(task_paths(&groups[1]), vec!["/b.parquet", "/c.parquet"]);
        assert_eq!(group_bytes(&groups), vec![110, 110]);
    }

    #[test]
    fn test_size_based_grouping_equal_sizes_spreads_like_round_robin() {
        let tasks = (0..4)
            .map(|i| sized_task(&format!("/{i}.parquet"), 50))
            .collect::<Vec<_>>();

        let groups = group_file_scan_tasks_by_size(tasks, 2);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 2);
        assert_eq!(groups[1].len(), 2);
    }

    #[test]
    fn test_size_based_grouping_zero_sizes_does_not_collapse() {
        let tasks = (0..6)
            .map(|i| sized_task(&format!("/{i}.parquet"), 0))
            .collect::<Vec<_>>();

        let groups = group_file_scan_tasks_by_size(tasks, 3);

        // The 1-byte weight floor keeps the tasks spread across all groups
        // instead of piling into the first one.
        assert_eq!(groups.len(), 3);
        for group in &groups {
            assert_eq!(group.len(), 2);
        }
    }

    #[test]
    fn test_size_based_grouping_fewer_tasks_than_partitions() {
        let tasks = vec![sized_task("/a.parquet", 10), sized_task("/b.parquet", 20)];

        let groups = group_file_scan_tasks_by_size(tasks, 4);

        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_size_based_grouping_empty_tasks() {
        let groups = group_file_scan_tasks_by_size(vec![], 4);

        assert_eq!(groups.len(), 1);
        assert!(groups[0].is_empty());
    }
}
