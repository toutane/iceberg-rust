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

use std::fmt::{Formatter, Result as FmtResult};
use std::pin::Pin;
use std::sync::Arc;
use std::vec;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::error::Result as DFResult;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use futures::{Stream, TryStreamExt};
use iceberg::expr::Predicate;
use iceberg::scan::{FileScanTask, FileScanTaskStream};
use iceberg::table::Table;

use super::scan_planning::{IcebergScanConfig, build_table_scan};
use crate::to_datafusion_error;

const DEFAULT_FILE_GROUP_DISPLAY_LIMIT: usize = 5;
const DEFAULT_FILE_DISPLAY_LIMIT: usize = 5;

/// Displays Iceberg file scan tasks using DataFusion's `FileGroupsDisplay`
/// shape and truncation rules, enriched with Iceberg planning metadata.
///
/// The byte fields are planned values, not runtime I/O counters:
///
/// - `scan_bytes` is the length of a task's planned byte range.
/// - `planned_scan_bytes` is the sum of `scan_bytes` for a group or the
///   complete scan.
/// - `file_size_bytes` is the size of the complete data file recorded in
///   Iceberg metadata. It can be larger than `scan_bytes` when a task scans
///   only part of a file, and can be repeated when several tasks reference
///   different ranges of the same file.
///
/// Actual bytes read at runtime can differ from these values because of file
/// format metadata reads, pruning, caching, and reader access patterns.
#[derive(Debug)]
struct FileTaskGroupsDisplay<'a>(&'a [Arc<[FileScanTask]>]);

impl DisplayAs for FileTaskGroupsDisplay<'_> {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> FmtResult {
        let group_count = self.0.len();
        let group_label = if group_count == 1 { "group" } else { "groups" };
        let planned_scan_bytes = self
            .0
            .iter()
            .flat_map(|group| group.iter())
            .map(|task| u128::from(task.length))
            .sum::<u128>();
        write!(f, "{{{group_count} {group_label}: [")?;

        let group_limit = match t {
            DisplayFormatType::Verbose => usize::MAX,
            DisplayFormatType::Default | DisplayFormatType::TreeRender => {
                DEFAULT_FILE_GROUP_DISPLAY_LIMIT
            }
        };
        for (group_index, group) in self.0.iter().take(group_limit).enumerate() {
            if group_index == 0 {
                writeln!(f)?;
            } else {
                writeln!(f, ",")?;
            }
            write!(f, "  ")?;
            FileTaskGroupDisplay(group).fmt_as(t, f)?;
        }
        if group_count > group_limit {
            if group_limit > 0 {
                writeln!(f, ",")?;
            } else {
                writeln!(f)?;
            }
            write!(f, "  ...")?;
        }
        if group_count > 0 {
            writeln!(f)?;
        }
        write!(f, "]}}, planned_scan_bytes={planned_scan_bytes}")
    }
}

#[derive(Debug)]
struct FileTaskGroupDisplay<'a>(&'a [FileScanTask]);

impl DisplayAs for FileTaskGroupDisplay<'_> {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> FmtResult {
        let planned_scan_bytes = self
            .0
            .iter()
            .map(|task| u128::from(task.length))
            .sum::<u128>();
        write!(f, "{{files=[")?;
        let file_limit = match t {
            DisplayFormatType::Verbose => usize::MAX,
            DisplayFormatType::Default | DisplayFormatType::TreeRender => {
                DEFAULT_FILE_DISPLAY_LIMIT
            }
        };
        for (file_index, task) in self.0.iter().take(file_limit).enumerate() {
            if file_index == 0 {
                writeln!(f)?;
            } else {
                writeln!(f, ",")?;
            }
            write!(f, "    ")?;
            FileTaskDisplay(task).fmt_as(t, f)?;
        }
        if self.0.len() > file_limit {
            if file_limit > 0 {
                writeln!(f, ",")?;
            } else {
                writeln!(f)?;
            }
            write!(f, "    ...")?;
        }
        if !self.0.is_empty() {
            writeln!(f)?;
            write!(f, "  ")?;
        }
        write!(f, "], planned_scan_bytes={planned_scan_bytes}}}")
    }
}

#[derive(Debug)]
struct FileTaskDisplay<'a>(&'a FileScanTask);

impl DisplayAs for FileTaskDisplay<'_> {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> FmtResult {
        let task = self.0;
        let end = task.start.saturating_add(task.length);
        write!(
            f,
            "{{path={}, range={}..{}, scan_bytes={}, file_size_bytes={}}}",
            task.data_file_path, task.start, end, task.length, task.file_size_in_bytes,
        )
    }
}

/// Manages the scanning process of an Iceberg [`Table`], encapsulating the
/// necessary details and computed properties required for execution planning.
#[derive(Debug)]
pub struct IcebergTableScan {
    /// A table in the catalog.
    table: Table,
    /// Snapshot, projection, output schema, and pushed predicates for this scan.
    scan_config: IcebergScanConfig,
    /// Stores certain, often expensive to compute,
    /// plan properties used in query optimization.
    plan_properties: Arc<PlanProperties>,
    /// Optional limit on the number of rows to return
    limit: Option<usize>,
    /// Pre-planned file scan tasks, grouped by partition. `None` keeps planning lazy.
    file_task_groups: Option<Vec<Arc<[FileScanTask]>>>,
}

impl IcebergTableScan {
    pub fn new(
        table: Table,
        scan_config: IcebergScanConfig,
        limit: Option<usize>,
        file_task_groups: Option<Vec<Vec<FileScanTask>>>,
    ) -> Self {
        let partition_count = file_task_groups.as_ref().map_or(1, |groups| groups.len());
        let plan_properties =
            IcebergTableScan::compute_properties(scan_config.output_schema(), partition_count);
        let file_task_groups = file_task_groups.map(|groups| {
            groups
                .into_iter()
                .map(Arc::<[FileScanTask]>::from)
                .collect()
        });

        Self {
            table,
            scan_config,
            plan_properties,
            limit,
            file_task_groups,
        }
    }

    pub fn table(&self) -> &Table {
        &self.table
    }

    pub fn snapshot_id(&self) -> Option<i64> {
        self.scan_config.snapshot_id()
    }

    pub fn projection(&self) -> Option<&[String]> {
        self.scan_config.column_names()
    }

    pub fn predicates(&self) -> Option<&Predicate> {
        self.scan_config.predicates()
    }

    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    pub fn scan_config(&self) -> &IcebergScanConfig {
        &self.scan_config
    }

    pub fn file_task_groups(&self) -> Option<&[Arc<[FileScanTask]>]> {
        self.file_task_groups.as_deref()
    }

    /// Computes [`PlanProperties`] used in query optimization.
    fn compute_properties(schema: ArrowSchemaRef, partition_count: usize) -> Arc<PlanProperties> {
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(partition_count),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ))
    }
}

impl ExecutionPlan for IcebergTableScan {
    fn name(&self) -> &str {
        "IcebergTableScan"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan + 'static>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(datafusion::common::DataFusionError::Internal(
                "IcebergTableScan is a leaf node and cannot have children".to_string(),
            ));
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
        let stream: Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>> = match &self
            .file_task_groups
        {
            Some(file_task_groups) => {
                let Some(file_task_group) = file_task_groups.get(partition).cloned() else {
                    return Err(datafusion::common::DataFusionError::Internal(format!(
                        "IcebergTableScan partition {partition} does not exist; scan has {} partitions",
                        file_task_groups.len()
                    )));
                };

                let tasks: FileScanTaskStream = Box::pin(futures::stream::iter(
                    (0..file_task_group.len()).map(move |idx| Ok(file_task_group[idx].clone())),
                ));
                let stream = build_table_scan(&self.table, &self.scan_config)?
                    .arrow_reader_builder()
                    // Eager planning lets DataFusion drive scan concurrency via output
                    // partitions. Match DataFusion's FileStream model, where each
                    // output partition owns one ScanState; keep one data file in
                    // flight per output partition here.
                    // https://github.com/apache/datafusion/blob/ad8e7b7f2babe3fcddc3a4f9b5cd1ac0d1b16ad9/datafusion/datasource/src/file_stream/scan_state.rs#L42-L43
                    .with_data_file_concurrency_limit(1)
                    .build()
                    // TODO: Avoid cloning FileScanTasks here once ArrowReader can accept shared tasks.
                    .read(tasks)
                    .map_err(to_datafusion_error)?
                    .stream()
                    .map_err(to_datafusion_error);

                Box::pin(stream)
            }
            None => {
                let table = self.table.clone();
                let scan_config = self.scan_config.clone();
                let fut = async move {
                    let table_scan = build_table_scan(&table, &scan_config)?;
                    let stream = table_scan
                        .to_arrow()
                        .await
                        .map_err(to_datafusion_error)?
                        .map_err(to_datafusion_error);
                    Ok::<_, datafusion::common::DataFusionError>(stream)
                };

                Box::pin(futures::stream::once(fut).try_flatten())
            }
        };

        // Apply a scan-partition bound if specified. In eager planning this is only
        // a per-partition bound; DataFusion's GlobalLimitExec remains responsible
        // for enforcing the final global limit.
        let limited_stream: Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>> =
            if let Some(limit) = self.limit {
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
            } else {
                Box::pin(stream)
            };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            limited_stream,
        )))
    }
}

impl DisplayAs for IcebergTableScan {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> FmtResult {
        write!(
            f,
            "IcebergTableScan projection:[{}] predicate:[{}]",
            self.projection().map_or(String::new(), |v| v.join(",")),
            self.predicates()
                .map_or(String::from(""), |p| format!("{p}")),
        )?;
        if let Some(file_task_groups) = &self.file_task_groups {
            write!(f, " file_groups=")?;
            FileTaskGroupsDisplay(file_task_groups).fmt_as(_t, f)?;
        }
        if let Some(limit) = self.limit {
            write!(f, " limit:[{limit}]")?;
        }
        Ok(())
    }
}
