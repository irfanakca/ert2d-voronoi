# ert2d – 2D DC resistivity forward modelling and inversion with Voronoi cells

`ert2d` solves the 2.5-D DC resistivity forward problem with finite elements and inverts apparent-resistivity data
with an Occam-type Gauss–Newton scheme. The model can be parameterized with

* **Voronoi cells** derived from the finite-element mesh (the mesh nodes are the generators; the cell boundaries are
  triangulated again to give the forward mesh),
* **triangles** (every element of the mesh is a parameter), or
* a **rectangular block grid**,

all inverted with the same forward solver, Jacobian, inversion driver and smoothness functional
(`ERT_REG=energy`: a finite-volume gradient energy `mᵀLm` with link weights `L_ij / d_ij`).
The code accompanies the article *A Voronoi parameterization derived from the finite-element mesh for 2D DC resistivity
inversion* (Akca and Öztürk Akca).

## Install

Python ≥ 3.10.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"        # Windows: .venv\Scripts\pip install -e ".[dev]"
python -m pytest -q
```

## Quick start

```python
from ert2d.io.dat_reader import read_dat       # RES2DINV-style .dat
from ert2d.mesh import build_param_mesh
from ert2d.inversion import invert

survey = read_dat("sample_data/synthetic_ws.dat")
pm = build_param_mesh(survey, kind="voronoi")  # or "triangle", "rect"
result = invert(survey, pm, max_iter=12)
print(result.misfit)
```

Forward modelling and inversion of a synthetic model with all three parameterizations:

```bash
python examples/01_forward_and_inversion.py
```

## Reproducing the tests of the article

The scripts are in `scripts/` (set `ERT_REG=energy` and `ERT_CORE_RULE=intersect`, as the scripts do by default):

| Script | Content |
| --- | --- |
| `model_suite.py` | models with discontinuities, inclined interfaces, irregular bodies, small objects; equal-count and natural-setting comparisons; mesh-refinement runs (`--converge`) |
| `adaptive_refine.py` | local refinement of the cells driven by the recovered gradients |
| `prior_boundary.py` | known boundaries: reduced smoothing weights across them, cells aligned with them |
| `embedded_model_test.py` | layer with embedded bodies |
| `operator_isotropy2.py` | directional behavior (anisotropy index) of the smoothness operators |
| `make_fig_v16.py` | figures of the tests |

Example: `python scripts/model_suite.py --models step basin blob arch --mode equal free --seeds 42 1 2 3 4 --out results/suite`.

## Layout

```
ert2d/io          data reader (.dat), results
ert2d/mesh        domain, triangulation (Triangle), Voronoi / triangular / rectangular parameterizations
ert2d/fem         shape functions, assembly, forward solver, sensitivity (Jacobian by reciprocity)
ert2d/inversion   data weights, smoothness operators, Occam update, iteration driver
ert2d/viz         sections and diagnostic figures
ert2d/gui         optional PySide6 interface (pip install -e ".[gui]")
scripts/          experiments of the article
sample_data/      small synthetic data sets
tests/            unit tests
```

## License

MIT (see `LICENSE`). If you use the code, please cite the article (see `CITATION.cff`).
