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



### Preconditions and numerical interpretation

The a posteriori `ratio_upper` applies to any valid sketch when its optimal-radius
cache was generated for the same data and sketch size. Data fingerprints reject
foreign and legacy caches. If the lower bound is zero, the ratio is infinity
for a positive radius and NaN when both are zero.

The pool-based `theory_bound = 2*opt_upper + 3*rho` requires an unmixed FFT
refinement over the pool. Direct `certificate` calls leave this field and
`bound_slack` as NaN unless `assume_fft_refinement=True` is explicitly asserted.
The assertion is the caller's responsibility; `sketch` sets it only when its
configuration satisfies the precondition. The sketch must belong to the pool.
These computations use floating-point Euclidean distances, not formally
rounded interval arithmetic.
