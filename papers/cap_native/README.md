# Cap-Native Architecture

Planned follow-up exploring the cap primitive as the foundational
architecture choice rather than an input augmentation: per-cap parameter
stacks at every block (attention QKV/O, MoE experts, RMSNorm gains,
output projection), with cap activations from layer 0 driving routing
at every layer.

Code is in `src/aware/cap_native/`. Bench harness:
`examples/cap_native_run_benchmark.rs`. Pre-set configs:
`scripts/run_multi_seed.sh cap_native_*`.

Experimental status: code complete; multi-seed runs pending.
