drop view if exists ml_training_dataset;
create view ml_training_dataset as (
with base as (
select
sql_text
, target_work_mem_mb
, target_max_parallel_workers_per_gather
, target_jit
, target_enable_indexscan
, target_enable_seqscan
, target_enable_bitmapscan
, target_enable_hashjoin
, target_enable_mergejoin
, target_enable_nestloop
, metric_planner_error_ratio as target_planner_error_ratio
, max_mem_limit_dop0 as target_mem_limit_dop0
, feature_total_cost
, feature_plan_rows
, feature_plan_width
, feature_num_joins
, feature_num_scans
, feature_num_aggs
, feature_num_sorts
, feature_num_filters
, feature_num_index_scans
, feature_num_mem_nodes
, feature_total_scan_size_bytes
, feature_max_node_cost
, metric_duration_ms
, metric_actual_rows
, metric_temp_written_blocks
, metric_shared_hit_blocks
, metric_shared_read_blocks
, metric_peak_memory_mb
from ml_results_optimal
union all
select
sql_text
, target_work_mem_mb
, target_max_parallel_workers_per_gather
, target_jit
, target_enable_indexscan
, target_enable_seqscan
, target_enable_bitmapscan
, target_enable_hashjoin
, target_enable_mergejoin
, target_enable_nestloop
, metric_planner_error_ratio as target_planner_error_ratio
, max_mem_limit_dop0 as target_mem_limit_dop0
, feature_total_cost
, feature_plan_rows
, feature_plan_width
, feature_num_joins
, feature_num_scans
, feature_num_aggs
, feature_num_sorts
, feature_num_filters
, feature_num_index_scans
, feature_num_mem_nodes
, feature_total_scan_size_bytes
, feature_max_node_cost
, metric_duration_ms
, metric_actual_rows
, metric_temp_written_blocks
, metric_shared_hit_blocks
, metric_shared_read_blocks
, metric_peak_memory_mb
from ml_results_failures
where sql_text not in (select distinct sql_text from ml_results_optimal)
)
, predata as (
select
b.*,
row_number() over (partition by
feature_total_cost
, feature_plan_rows
, feature_plan_width
, feature_num_joins
, feature_num_scans
, feature_num_aggs
, feature_num_sorts
, feature_num_filters
, feature_num_index_scans
, feature_num_mem_nodes
, feature_total_scan_size_bytes
, feature_max_node_cost
order by metric_duration_ms asc) as rn
from base b
)
select
target_work_mem_mb
, target_max_parallel_workers_per_gather
, target_jit
, target_enable_indexscan
, target_enable_seqscan
, target_enable_bitmapscan
, target_enable_hashjoin
, target_enable_mergejoin
, target_enable_nestloop
, target_planner_error_ratio
, target_mem_limit_dop0
, feature_total_cost
, feature_plan_rows
, feature_plan_width
, feature_num_joins
, feature_num_scans
, feature_num_aggs
, feature_num_sorts
, feature_num_filters
, feature_num_index_scans
, feature_num_mem_nodes
, feature_total_scan_size_bytes
, feature_max_node_cost
, metric_duration_ms
, metric_actual_rows
, metric_temp_written_blocks
, metric_shared_hit_blocks
, metric_shared_read_blocks
, metric_peak_memory_mb
from predata
where rn = 1);