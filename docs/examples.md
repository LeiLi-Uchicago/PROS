# Examples

The `examples/` directory contains small runnable scripts:

- `01_basic_numpy.py`: basic matrix sketching.
- `02_ann_data.py`: AnnData workflow.
- `03_certification.py`: certificate fields.
- `04_compare_configs.py`: simple configuration comparison.

Run an example from the repository root:

```bash
python examples/01_basic_numpy.py
```

The checkout examples use `partitioner="pc_tree"` to keep their behaviour
simple. The installed package includes `scikit-learn`, which is required by
the default `partitioner="kmeans"` workflow.
