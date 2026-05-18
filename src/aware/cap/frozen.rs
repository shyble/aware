// Frozen-random cap variant: keys are random unit vectors set at init,
// then NEVER changed (no gradient, no audit). Reservoir-computing style.
// Identity of each cap is rock-solid: cap_47 always has the same key.
// Downstream edges must learn what to do with the random features.
