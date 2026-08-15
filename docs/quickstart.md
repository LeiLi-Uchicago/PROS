# Quickstart

```python
import numpy as np
from pros import sketch

rng = np.random.default_rng(0)
X = rng.random((5000, 50))

res = sketch(X, n=500, seed=0)
X_sketch = X[res.indices]
```

The result is dict-like and also supports attribute access:

```python
print(res["indices"])
print(res.indices)
print(res.timings)
```

For AnnData:

```python
from pros import sketch_adata

res = sketch_adata(adata, n=5000, use_rep="X_pca")
subset = adata[res.indices]
```

