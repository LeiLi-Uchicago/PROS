# Installation

Install from a local checkout:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[adata]"
python -m pip install -e ".[dev]"
python -m pip install -e ".[docs]"
```

The core package depends on `numpy` and `scikit-learn`. AnnData and Scanpy are
optional so matrix-only users do not need the single-cell stack.

