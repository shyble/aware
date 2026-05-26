# Concept Layer

Planned work on a concept layer placed in parallel to the prediction
pipeline: window=1 caps with frozen-identity keys (KMeans centroids)
and gradient-trained values, with sparse top-K activation. The
architectural claim is that concept IDs persist across substrate
retraining, providing cognition-level stable identifiers above the
drifting substrate.

Code is in `src/aware/concept_layer.rs`. Working example:
`examples/train_with_concepts.rs`.

Experimental status: code complete; stability-across-substrate-retraining
experiments pending.
