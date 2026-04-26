drop view if exists ml_training_dataset;
CREATE view ml_training_dataset AS
SELECT distinct
    query_id
    , sf
    , score
    , reason
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
    , target_work_mem_mb
    , target_max_parallel_workers_per_gather
    , target_jit
    , target_enable_indexscan
    , target_enable_seqscan
    , target_enable_bitmapscan
    , target_enable_hashjoin
    , target_enable_mergejoin
    , target_enable_nestloop
    , metric_planner_error_ratio
    , max_mem_limit_dop0
    , metric_duration_ms
    , metric_actual_rows
    , metric_temp_written_blocks
    , metric_shared_hit_blocks
    , metric_shared_read_blocks
    , metric_peak_memory_mb
FROM ml_results_optimal;