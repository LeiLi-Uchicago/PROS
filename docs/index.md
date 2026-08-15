# PROS

PROS builds diversity-preserving sketches of large coordinate matrices. It
first partitions the data, oversamples a candidate pool inside those blocks,
and then refines globally over the pool.

The returned sketch is an array of row indices into the original matrix. For
evaluation runs, PROS can also attach a certificate with the sketch covering
radius, pool covering radius, and a proven approximation-ratio upper bound.

