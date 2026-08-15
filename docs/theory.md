# Theory Notes

PROS targets the k-center covering objective:

```text
R(X, S) = max_i min_s d(x_i, s)
```

The stage-1 pool has covering radius `rho = R(X, P)`. Stage 2 refines over the
pool instead of the full data, so its cost scales with the pool size rather
than the full population size.

When `certify=True`, PROS computes:

- `radius`: `R(X, S)`, the covering radius of the returned sketch.
- `rho`: `R(X, P)`, the covering radius of the candidate pool.
- `opt_lower`: a valid lower bound on the optimal `n`-center radius.
- `ratio_upper`: `radius / opt_lower`, a valid upper bound on the true
  approximation ratio.

Certification requires extra full-data passes and is best treated as an
evaluation instrument rather than part of the production fast path.

