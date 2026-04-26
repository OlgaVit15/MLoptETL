drop view if exists ml_training_dataset;
create view ml_training_dataset as (
with base as (
select
sql_text
, reason as target_reason
, max_mem_limit_dop0 as target_mem_limit_dop0
, target_mt_dop
, target_mem_limit
, target_num_scanner_threads
, target_default_join_distribution_mode
, target_disable_codegen
, feature_plan_num_joins
, feature_plan_num_broadcast_joins
, feature_plan_num_scan_nodes
, feature_plan_num_agg_nodes
, feature_plan_num_files
, feature_plan_has_missing_stats
, feature_plan_total_scan_size_bytes
, feature_plan_max_cardinality
, feature_plan_max_row_size
, metric_duration_ms
, metric_pmu
, metric_spill_bytes
from ml_results_optimal
union all
select
sql_text
, substring(reason from 8) as target_reason
, max_mem_limit_dop0 as target_mem_limit_dop0
, target_mt_dop
, target_mem_limit
, target_num_scanner_threads
, target_default_join_distribution_mode
, target_disable_codegen
, feature_plan_num_joins
, feature_plan_num_broadcast_joins
, feature_plan_num_scan_nodes
, feature_plan_num_agg_nodes
, feature_plan_num_files
, feature_plan_has_missing_stats
, feature_plan_total_scan_size_bytes
, feature_plan_max_cardinality
, feature_plan_max_row_size
, metric_duration_ms
, metric_pmu
, metric_spill_bytes
from ml_results_failures
where sql_text not in (select distinct sql_text from ml_results_optimal)
)
, predata as (
select
b.*,
row_number() over (partition by sql_text order by metric_duration_ms asc) as rn
from base b
)
select
target_reason
, target_mem_limit_dop0
, target_mt_dop
, target_mem_limit
, target_num_scanner_threads
, target_default_join_distribution_mode
, target_disable_codegen
, feature_plan_num_joins
, feature_plan_num_broadcast_joins
, feature_plan_num_scan_nodes
, feature_plan_num_agg_nodes
, feature_plan_num_files
, feature_plan_has_missing_stats
, feature_plan_total_scan_size_bytes
, feature_plan_max_cardinality
, feature_plan_max_row_size
, metric_duration_ms
, metric_pmu
, metric_spill_bytes
from predata
where rn = 1);