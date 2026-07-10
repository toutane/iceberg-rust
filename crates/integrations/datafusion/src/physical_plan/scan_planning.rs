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

use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::error::Result as DFResult;
use datafusion::prelude::Expr;
use futures::TryStreamExt;
use iceberg::expr::Predicate;
use iceberg::scan::{FileScanTask, TableScan};
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
) -> DFResult<Vec<Vec<FileScanTask>>> {
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

    Ok(group_file_scan_tasks_by_size(tasks, target_partitions))
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
