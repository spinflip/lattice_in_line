# vmps_geometry

Geometry helpers, cluster graph tables, optimized permutations, supersite
assignments, and unit-cell generators used by `vmps_torch`.

## Editable Development

This repository is intended to live next to `vmps_torch`:

```bash
cd /Users/rr/VMPSbundle/vmps_geometry
pip install -e .

cd /Users/rr/VMPSbundle/vmps_torch
pip install -e .
```

The import package is `vmps_geometry`.

## Compatibility

`vmps_torch` keeps a temporary `vmps_torch.geometry` compatibility namespace
that re-exports this package, so existing scripts can migrate gradually.
