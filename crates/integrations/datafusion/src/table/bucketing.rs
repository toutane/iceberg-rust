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

use datafusion::arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Float32Array, Float64Array, Int32Array, Int64Array,
    StringArray,
};
use datafusion::arrow::datatypes::{DataType, Schema as ArrowSchema};
use datafusion::common::hash_utils::create_hashes;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::repartition::REPARTITION_RANDOM_STATE;
use iceberg::scan::FileScanTask;
use iceberg::spec::{Literal, PrimitiveLiteral, Transform};
use iceberg::table::Table;

/// Identity-partitioned column that is also present in the output projection
/// and whose Arrow type can be reconstructed from a `Literal` for hashing.
pub(super) struct IdentityCol {
    pub(super) name: String,
    /// Position of this column in the *output* schema (after projection).
    pub(super) output_idx: usize,
    /// Position of this column inside the partition spec's `fields()` slice,
    /// matching the slot order of `FileScanTask::partition`.
    pub(super) spec_field_idx: usize,
    pub(super) output_dtype: DataType,
}

/// Inspect the table's default partition spec and return the list of identity
/// columns that can support a [`Partitioning::Hash`] declaration. Returns
/// `None` if any condition is violated:
///   - the source column for an identity field is not in the output projection
///   - the source column's Arrow type is not currently supported by
///     [`literal_to_array`]
///   - the table has spec evolution (>1 historical specs), since older files
///     may carry a partition tuple that does not align with the default spec
///
/// Returning `None` forces the scan to declare `UnknownPartitioning` even if
/// bucketing succeeds.
pub(super) fn compute_identity_cols(
    table: &Table,
    output_schema: &ArrowSchema,
) -> Option<Vec<IdentityCol>> {
    let metadata = table.metadata();
    if metadata.partition_specs_iter().len() > 1 {
        return None;
    }
    let spec = metadata.default_partition_spec();
    let table_schema = metadata.current_schema();

    let mut cols = Vec::new();
    for (spec_field_idx, pf) in spec.fields().iter().enumerate() {
        if pf.transform != Transform::Identity {
            continue;
        }
        let source_field = table_schema.field_by_id(pf.source_id)?;
        let output_idx = output_schema.index_of(source_field.name.as_str()).ok()?;
        let output_dtype = output_schema.field(output_idx).data_type().clone();
        if !is_supported_dtype(&output_dtype) {
            return None;
        }
        cols.push(IdentityCol {
            name: source_field.name.clone(),
            output_idx,
            spec_field_idx,
            output_dtype,
        });
    }
    Some(cols)
}

fn is_supported_dtype(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Boolean
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::Date32
    )
}

/// Spec field with `Transform::Bucket(_)`. The source column must be in the
/// output projection so we can reference it via `Column` in `Partitioning::Hash`.
/// We don't need the Arrow type because the partition tuple slot for a bucket
/// transform is always `Int32` (the spec-defined `result_type`).
pub(super) struct BucketCol {
    pub(super) name: String,
    /// Position of this column in the *output* schema (after projection).
    pub(super) output_idx: usize,
    /// Position of this column inside the partition spec's `fields()` slice,
    /// matching the slot order of `FileScanTask::partition`.
    pub(super) spec_field_idx: usize,
}

/// Inspect the table's default partition spec and return the bucket column
/// usable for a [`Partitioning::Hash`] declaration. The spec must contain
/// exactly one `Transform::Bucket(_)` field, and its source column must
/// be present in the output projection. Returns `None` on spec evolution,
/// empty spec, multi-field specs (even when all fields are bucket
/// transforms), non-bucket transforms, or when the bucket source column is
/// absent from the projection.
///
/// This deliberately rejects mixed identity+bucket specs: those are handled
/// by [`compute_identity_cols`] which retains only the identity fields.
pub(super) fn compute_bucket_cols(
    table: &Table,
    output_schema: &ArrowSchema,
) -> Option<Vec<BucketCol>> {
    let metadata = table.metadata();
    // TODO: extend to handle time travel, e.g. when a `snapshot_id` is provided,
    // we could accept tables with spec evolution as long as every file
    // reachable from that snapshot was written under the same partition spec
    // as the one we declare on.
    if metadata.partition_specs_iter().len() > 1 {
        return None;
    }
    let spec = metadata.default_partition_spec();
    let fields = spec.fields();
    if fields.len() != 1 {
        return None;
    }
    let pf = &fields[0];
    if !matches!(pf.transform, Transform::Bucket(_)) {
        return None;
    }
    let table_schema = metadata.current_schema();
    let source_field = table_schema.field_by_id(pf.source_id)?;
    let output_idx = output_schema.index_of(source_field.name.as_str()).ok()?;
    Some(vec![BucketCol {
        name: source_field.name.clone(),
        output_idx,
        spec_field_idx: 0,
    }])
}

/// Single-entry partition-key descriptor used by [`bucket_tasks`] and
/// `IcebergTableProvider::scan` to drive both task distribution and the
/// `Partitioning::Hash` declaration.
pub(super) enum PartitionKeys {
    Identity(Vec<IdentityCol>),
    Bucket(Vec<BucketCol>),
}

impl PartitionKeys {
    /// `Column` exprs (one per key column) referencing the *output* schema,
    /// suitable for `Partitioning::Hash`.
    pub(super) fn column_exprs(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        match self {
            PartitionKeys::Identity(cols) => cols
                .iter()
                .map(|c| Arc::new(Column::new(&c.name, c.output_idx)) as Arc<dyn PhysicalExpr>)
                .collect(),
            PartitionKeys::Bucket(cols) => cols
                .iter()
                .map(|c| Arc::new(Column::new(&c.name, c.output_idx)) as Arc<dyn PhysicalExpr>)
                .collect(),
        }
    }
}

/// Return the partition keys that drive a `Partitioning::Hash` declaration,
/// or `None` if the spec is not hash-declarable.
///
/// Identity is tried first — it also covers mixed identity+bucket specs,
/// keeping only the identity fields. Otherwise we fall back to a
/// single-column pure-bucket spec.
pub(super) fn compute_partition_keys(
    table: &Table,
    output_schema: &ArrowSchema,
    identity_partitioning_enabled: bool,
    bucket_execution_enabled: bool,
) -> Option<PartitionKeys> {
    if identity_partitioning_enabled
        && let Some(cols) = compute_identity_cols(table, output_schema)
        && !cols.is_empty()
    {
        return Some(PartitionKeys::Identity(cols));
    }
    if bucket_execution_enabled {
        compute_bucket_cols(table, output_schema).map(PartitionKeys::Bucket)
    } else {
        None
    }
}

/// Group `tasks` into `n_partitions` buckets, one per DataFusion output
/// partition of the surrounding `IcebergTableScan`.
///
/// # Arguments
///
/// * `tasks` - every `FileScanTask` produced by `plan_files()` for the
///   current scan (already filtered by predicate and projection). They
///   will be consumed and redistributed across the returned buckets.
///
/// * `n_partitions` - number of output partitions to produce. The
///   surrounding scan passes `min(target_partitions, tasks.len()).max(1)`,
///   so this is always `>= 1` in practice. A value of `0` short-circuits
///   to an empty result.
///
/// * `keys` - describes how a task is assigned to a bucket. Three cases:
///   - `Some(PartitionKeys::Identity(cols))`: hash the task's identity
///     partition values via [`REPARTITION_RANDOM_STATE`] then take
///     `% n_partitions`. Using DataFusion's repartition hash state makes
///     the assignment match what `RepartitionExec(Hash([cols], n))` would
///     compute row-by-row, which lets the planner elide a downstream
///     repartition.
///   - `Some(PartitionKeys::Bucket(cols))`: read the task's pre-computed
///     bucket index `idx` and take `idx % n_partitions`.
///     value, so same-key rows are already co-located at the file level.
///   - `None`: the spec is not hash-declarable. Every task takes the
///     fallback path below.
///
/// # Fallback
///
/// When a task can't yield its key (missing partition tuple, null slot,
/// unsupported literal type), it is placed deterministically via
/// `fallback_hash(data_file_path) % n_partitions` and the second return
/// flag flips to `false`.
///
/// # Returns
///
/// * `Vec<Vec<FileScanTask>>` of length `n_partitions` - the tasks
///   regrouped by output partition (some buckets may be empty).
/// * `bool` - `true` iff every task supplied a full key. The caller uses
///   this to decide between `Partitioning::Hash` (when `true` and
///   `keys.is_some()`) and `Partitioning::UnknownPartitioning` (otherwise).
///   A single fallback occurrence is enough to break the
///   "same key -> same partition" contract, hence the all-or-nothing flag.
pub(super) fn bucket_tasks(
    tasks: Vec<FileScanTask>,
    n_partitions: usize,
    keys: Option<&PartitionKeys>,
) -> (Vec<Vec<FileScanTask>>, bool) {
    if n_partitions == 0 {
        return (Vec::new(), tasks.is_empty());
    }
    let mut buckets: Vec<Vec<FileScanTask>> = (0..n_partitions).map(|_| Vec::new()).collect();
    let mut all_full_key = true;

    for task in tasks {
        let dest = match keys {
            Some(PartitionKeys::Identity(cols)) => {
                identity_hash(&task, cols).map(|h| (h % n_partitions as u64) as usize)
            }
            Some(PartitionKeys::Bucket(cols)) => {
                bucket_index(&task, cols).map(|idx| (idx % n_partitions as u64) as usize)
            }
            None => None,
        };
        let dest = dest.unwrap_or_else(|| {
            all_full_key = false;
            fallback_hash(&task) as usize % n_partitions
        });
        buckets[dest].push(task);
    }
    (buckets, all_full_key)
}

/// Hash the identity-partition values of `task` using
/// [`REPARTITION_RANDOM_STATE`] so the bucket assignment matches DataFusion's
/// hash-repartition convention. Returns `None` if the task lacks partition
/// data or any required slot is null/unsupported.
fn identity_hash(task: &FileScanTask, cols: &[IdentityCol]) -> Option<u64> {
    if cols.is_empty() {
        return None;
    }
    let partition = task.partition.as_ref()?;
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(cols.len());
    for col in cols {
        let lit = partition.fields().get(col.spec_field_idx)?.as_ref()?;
        arrays.push(literal_to_array(lit, &col.output_dtype)?);
    }
    let mut hashes = vec![0u64; 1];
    create_hashes(
        &arrays,
        REPARTITION_RANDOM_STATE.random_state(),
        &mut hashes,
    )
    .ok()?;
    Some(hashes[0])
}

/// Return the bucket index of `task` as a `u64`. The slot for a
/// `Transform::Bucket(_)` field is always `Int32` per the Iceberg spec; a
/// missing slot or non-`Int` literal returns `None`, driving the caller's
/// `all_full_key` flag to `false`.
fn bucket_index(task: &FileScanTask, cols: &[BucketCol]) -> Option<u64> {
    let col = cols.first()?;
    let partition = task.partition.as_ref()?;
    let lit = partition.fields().get(col.spec_field_idx)?.as_ref()?;
    let idx = match lit {
        Literal::Primitive(PrimitiveLiteral::Int(v)) => *v,
        _ => return None,
    };
    // `idx` is non-negative per the Iceberg spec (result of `bucket[N]` is
    // in `[0, N)`); cast through `u32` first to avoid sign extension if the
    // file accidentally carries a negative slot.
    Some(idx as u32 as u64)
}

/// Deterministic per-file fallback used when `identity_hash` cannot produce a
/// bucket. The hash function does not need to match DataFusion's because any
/// task taking this path causes the scan to drop to `UnknownPartitioning`.
fn fallback_hash(task: &FileScanTask) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    task.data_file_path.hash(&mut hasher);
    hasher.finish()
}

/// Materialize a single-element Arrow array of `dt` holding the value of
/// `lit`. The Arrow type must match what DataFusion will see for this column
/// at scan time, otherwise `create_hashes` would dispatch on a different type
/// and produce a hash that disagrees with DataFusion's row-wise hashing.
fn literal_to_array(lit: &Literal, dt: &DataType) -> Option<ArrayRef> {
    let prim = match lit {
        Literal::Primitive(p) => p,
        _ => return None,
    };
    Some(match (prim, dt) {
        (PrimitiveLiteral::Boolean(v), DataType::Boolean) => Arc::new(BooleanArray::from(vec![*v])),
        (PrimitiveLiteral::Int(v), DataType::Int32) => Arc::new(Int32Array::from(vec![*v])),
        (PrimitiveLiteral::Int(v), DataType::Date32) => Arc::new(Date32Array::from(vec![*v])),
        (PrimitiveLiteral::Long(v), DataType::Int64) => Arc::new(Int64Array::from(vec![*v])),
        (PrimitiveLiteral::Float(v), DataType::Float32) => Arc::new(Float32Array::from(vec![v.0])),
        (PrimitiveLiteral::Double(v), DataType::Float64) => Arc::new(Float64Array::from(vec![v.0])),
        (PrimitiveLiteral::String(v), DataType::Utf8) => {
            Arc::new(StringArray::from(vec![v.as_str()]))
        }
        _ => return None,
    })
}
