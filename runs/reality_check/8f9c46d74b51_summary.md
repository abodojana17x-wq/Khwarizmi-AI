# Architecture Reality Check Report
**Run ID:** 8f9c46d74b51
**Timestamp:** 2026-08-31T16:43:38.206569
**Git Commit:** f328eec4ea2b546b7da875202960966885470072
**Branch:** qwen-code-32410811-4cc8-4fe0-a97c-a19bff7c45e4

## Summary
- Total Experiments: 7
- Passed: 5
- Failed: 1
- Errors: 1

## Experiment Results

### KSC_MEMORY_RETENTION
**Status:** PASS
**Metrics:**
- test_distances: [128, 256, 512, 1024, 2048, 4096, 8192]
- results_by_distance: {128: {'accuracy': 1.0, 'output_shape': [1, 128, 512], 'status': 'pass'}, 256: {'accuracy': 0.0, 'status': 'skipped_exceeds_max_seq_len'}, 512: {'accuracy': 0.0, 'status': 'skipped_exceeds_max_seq_len'}, 1024: {'accuracy': 0.0, 'status': 'skipped_exceeds_max_seq_len'}, 2048: {'accuracy': 0.0, 'status': 'skipped_exceeds_max_seq_len'}, 4096: {'accuracy': 0.0, 'status': 'skipped_exceeds_max_seq_len'}, 8192: {'accuracy': 0.0, 'status': 'skipped_exceeds_max_seq_len'}}
- max_seq_len_config: 128

### KSC_OVERWRITE_RESET
**Status:** PASS
**Metrics:**
- state_changed_on_overwrite: True
- retention_valid_old: True
- retention_valid_new: True
- delay_test_results: {1: {'finite': True, 'differs_from_initial': True}, 5: {'finite': True, 'differs_from_initial': True}, 10: {'finite': True, 'differs_from_initial': True}, 20: {'finite': True, 'differs_from_initial': True}}

### ARRC_COMPUTE_SAVINGS
**Status:** PASS
**Metrics:**
- fixed_compute_mean_cycles: 1.0
- adaptive_compute_mean_cycles: 2.75
- tokens_halted_early: 16
- tokens_at_max: 48
- halting_variance: 0.190476194024086
- ponder_loss_adaptive: 0.03345761075615883
- total_tokens: 64
- logical_compute_reduction: 0.08333333333333333
- physical_flops_measured: False
- note: Logical halting measured; physical FLOP savings require kernel-level profiling

### MOE_VS_DENSE
**Status:** PASS
**Metrics:**
- total_parameters: 66816
- expert_parameters: 66304
- router_parameters: 512
- active_parameters_per_token: 33664
- active_param_ratio: 0.5038314176245211
- num_experts_total: 4
- top_k: 2
- experts_executed_last_forward: 4
- expert_fractions: [0.4765625, 0.5, 0.5, 0.5234375]
- routing_aux_loss: 0.020043902099132538
- avg_sparse_latency_ms: 1.6722614999935104

### MEMORY_SUBSYSTEM_AUDIT
**Status:** ERROR
**Errors:**
- memory_table is missing required key 'keys'...
- Traceback (most recent call last):
  File "/workspace/benchmarks/reality_check/run_reality_check.py", line 562, in run_memory_subsystem_audit
    updated_table_3 = long_term_memory.forget(
           ...

### COGNITIVE_ROUTING_AUDIT
**Status:** PASS
**Metrics:**
- probs_shape_valid: True
- pathway_shape_valid: True
- pathways_in_range: True
- probs_sum_to_one: True
- is_deterministic: True
- has_pathway_costs: True
- router_sees_input: True
- input_sensitive: True
- num_pathways: 5
- pathway_names: ['FAST', 'CODING', 'REASONING', 'PROJECT_PLAN', 'VERIFICATION']

### TIER_DEFINITIONS_AUDIT
**Status:** FAIL
**Metrics:**
- available_tiers: ['TinyTest', 'Prototype', 'Prototype-50M', 'Prototype-50M_ERROR', 'Prototype-150M', 'Prototype-150M_ERROR', 'Small', 'Edge']
- nano_mobile_pro_ultra_present: False
- documented_tiers: ['Nano', 'Mobile', 'Pro', 'Ultra']
- implemented_tiers: ['TinyTest', 'Prototype', 'Small', 'Edge']
- documentation_gap: True
- tier_count: 6
**Failures:**
- Documented tiers (Nano/Mobile/Pro/Ultra) not found in implementation. Implementation uses TinyTest/Prototype/Small/Edge naming.
