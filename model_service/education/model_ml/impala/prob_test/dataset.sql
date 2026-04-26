drop view if exists ml_training_dataset;
CREATE view ml_training_dataset AS
WITH neg as  (
select query_id, sf, max(cast(substring(reason from 6 for 1) as int)) as run_id
from ml_results_failures
where reason not like '%error%' and reason not like '%spill%' and reason not like '%rejected_limit%'
group by query_id, sf
)
, all_runs AS (
    SELECT distinct
    query_id,
    sf,
    score,
    feature_plan_num_joins	,
    feature_plan_num_broadcast_joins	,
    feature_plan_num_scan_nodes	,
    feature_plan_num_agg_nodes	,
    feature_plan_num_files	,
    feature_plan_has_missing_stats	,
    feature_plan_total_scan_size_bytes	,
    feature_plan_max_cardinality	,
    feature_plan_max_row_size ,
    target_mem_limit,
    target_mt_dop,
    max_mem_limit_dop0,
    target_num_scanner_threads,
    target_default_join_distribution_mode,
    target_disable_codegen,
    metric_duration_ms,
    metric_pmu,
    metric_spill_bytes,
    1 as is_success FROM ml_results_optimal
    UNION ALL
    SELECT distinct
    b.query_id,
    b.sf,
    b.score,
    b.feature_plan_num_joins	,
    b.feature_plan_num_broadcast_joins	,
    b.feature_plan_num_scan_nodes	,
    b.feature_plan_num_agg_nodes	,
    b.feature_plan_num_files	,
    b.feature_plan_has_missing_stats	,
    b.feature_plan_total_scan_size_bytes	,
    b.feature_plan_max_cardinality	,
    b.feature_plan_max_row_size ,
    b.target_mem_limit,
    b.target_mt_dop,
    b.max_mem_limit_dop0,
    b.target_num_scanner_threads,
    b.target_default_join_distribution_mode,
    b.target_disable_codegen,
    b.metric_duration_ms,
    b.metric_pmu,
    b.metric_spill_bytes,
    0 as is_success
    FROM ml_results_failures b
    inner join neg n on n.query_id = b.query_id and n.sf = b.sf and n.run_id = cast(substring(b.reason from 6 for 1) as int)
    left join ml_results_optimal o on o.query_id = b.query_id and o.sf = b.sf
    where o.query_id is null and b.reason not like '%error%' and b.reason not like '%spill%' and b.reason not like '%rejected_limit%'
)
SELECT
    query_id,
    sf,
    is_success,
    score,
    feature_plan_num_joins	,
    feature_plan_num_broadcast_joins	,
    feature_plan_num_scan_nodes	,
    feature_plan_num_agg_nodes	,
    feature_plan_num_files	,
    feature_plan_has_missing_stats	,
    feature_plan_total_scan_size_bytes	,
    feature_plan_max_cardinality	,
    feature_plan_max_row_size ,
    target_mem_limit,
    target_mt_dop,
    max_mem_limit_dop0,
    target_num_scanner_threads,
    target_default_join_distribution_mode,
    target_disable_codegen,
    metric_duration_ms,
    metric_pmu,
    metric_spill_bytes
FROM all_runs;