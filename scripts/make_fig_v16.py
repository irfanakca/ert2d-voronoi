"""Figures of manuscript v16 (model suite, convergence, adaptive refinement, known boundaries).

    $env:ERT_REG="energy"; $env:ERT_CORE_RULE="intersect"; python scripts/make_fig_v16.py
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("ERT_REG", "energy")
os.environ.setdefault("ERT_CORE_RULE", "intersect")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import box

import model_suite as ms
from fair import footprint_polygon
from ert2d.io.dat_reader import read_dat
from ert2d.viz.sections import format_log_colorbar, model_section_from_arrays

# folder with the result folders written by the experiment scripts (default: <repository>/results)
ROOT = Path(os.environ.get("ERT_RESULTS_ROOT", Path(__file__).resolve().parents[1] / "results"))
R24 = ROOT / "results_v24"
OUT = R24 / "figs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13})
COL = {"voronoi": "C0", "triangle": "C1", "rect": "C2", "rect1": "C2", "rect2": "C3"}
LAB = {"voronoi": "Voronoi", "triangle": "Triangular", "rect": "Rectangular", "rect1": "Rectangular", "rect2": "Rectangular (2 per spacing)"}
TITLE = {"step": "Stepped basement and vertical contact", "basin": "Inclined interfaces", "blob": "Irregularly shaped bodies",
         "arch": "Wall, chamber and pipe", "emb": "Layer with embedded bodies"}


def _survey(model):
    spec = ms.SPECS[model]()
    return spec, read_dat(str(ms.geometry_file(spec)))


def _outlines(ax, spec):
    for _, _, poly in spec["shapes"]:
        xs, zs = poly.exterior.xy
        ax.plot(xs, zs, "k-", lw=0.9)


def true_panel(ax, spec, sv, lo, hi):
    ex = np.asarray(sv.electrode_x, float)
    zm = float(sv.z_max)
    X, Z = np.meshgrid(np.linspace(ex.min(), ex.max(), 500), np.linspace(-zm, 0, 200))
    ax.pcolormesh(X, Z, ms.true_log10(spec, X, Z), cmap="turbo", vmin=np.log10(lo), vmax=np.log10(hi), shading="auto")
    ax.plot(ex, np.zeros_like(ex), "v", color="0.1", ms=3.5, clip_on=False)
    ax.set_xlim(ex.min(), ex.max()); ax.set_ylim(-zm, 0)


def models_figure(pairs=((("step", "blob"), "fig_models_main.png"), (("basin", "arch"), "fig_models_supp.png"))):
    S = pickle.load(open(R24 / "suite" / "suite.pkl", "rb"))
    S["models"].update(pickle.load(open(R24 / "arch_fix" / "suite" / "suite.pkl", "rb"))["models"])     # arch rerun, correct spacing
    lims = {"step": (30, 400), "basin": (100, 500), "blob": (15, 800), "arch": (15, 5000)}
    for models, fname in pairs:
        fig, axes = plt.subplots(4, len(models), figsize=(6.9 * len(models), 6.6), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.02, hspace=0.02, w_pad=0.04)
        letters = iter("abcdefghijkl")
        for c, m in enumerate(models):
            spec, sv = _survey(m)
            ex = np.asarray(sv.electrode_x, float)
            zm = float(sv.z_max)
            fp = footprint_polygon(sv)
            lo, hi = lims[m]
            true_panel(axes[0, c], spec, sv, lo, hi)
            axes[0, c].set_title(f"({next(letters)}) True model: {TITLE[m].lower()}", loc="left", fontsize=12, fontweight="bold")
            mods = S["models"][(m, "equal", 0)]
            for r, kind in enumerate(("voronoi", "triangle", "rect1"), start=1):
                ax = axes[r, c]
                d = mods[kind]
                mpb = model_section_from_arrays(d["poly_verts"], d["cell_idx"], d["rho_cell"], ax=ax, electrodes=ex, depth_exaggeration=1.0,
                                                x_lim=(ex.min(), ex.max()), z_lim=zm, colorbar=False, clim=(lo, hi), title=None, clip_verts=fp, xlabel="")
                _outlines(ax, spec)
                ax.set_xlim(ex.min(), ex.max()); ax.set_ylim(-zm, 0)
                ax.set_title(f"({next(letters)}) {LAB[kind]}, {d['n']} unknowns", loc="left", fontsize=12, fontweight="bold")
            for ax in axes[:, c]:
                ax.set_aspect("equal", adjustable="box")
            axes[-1, c].set_xlabel("Distance (m)")
            cb = fig.colorbar(mpb, ax=axes[:, c].tolist(), location="right", fraction=0.04, pad=0.01)
            format_log_colorbar(cb); cb.set_label(r"$\rho$ ($\Omega\cdot$m)", fontsize=12)
        for ax in axes[:, 0]:
            ax.set_ylabel("Depth (m)")
        fig.savefig(OUT / fname, dpi=170)
        plt.close(fig)
    print("models")


def converge_figure(models=("step", "blob", "emb"), fname="fig_converge_main.png", shape=(1, 3), size=(11.5, 3.7)):
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    C = pickle.load(open(R24 / "converge" / "suite.pkl", "rb"))
    rows = [r for r in C["rows"] if r["model"] != "arch"] + pickle.load(open(R24 / "arch_fix" / "converge" / "suite.pkl", "rb"))["rows"]
    fig, axes = plt.subplots(*shape, figsize=size, layout="constrained", squeeze=False)
    short = {"step": "Steps, vertical contact", "basin": "Inclined interfaces", "blob": "Irregular bodies", "arch": "Wall, chamber, pipe",
             "emb": "Embedded bodies"}
    for ax, m in zip(axes.ravel(), models):
        for kind in ("voronoi", "triangle", "rect"):
            xs, ys, sd = [], [], []
            for lev in range(4):
                s = [r for r in rows if r["model"] == m and r["kind"] == kind and r["level"] == lev]
                if s:
                    xs.append(np.mean([r["n"] for r in s])); ys.append(np.mean([r["err"] for r in s]))
                    sd.append(np.std([r["err"] for r in s], ddof=1) if len(s) > 1 else 0.0)
            ax.errorbar(xs, ys, yerr=sd, color=COL[kind], marker="osv"[("voronoi", "triangle", "rect").index(kind)], ms=6, lw=1.5, capsize=2,
                        label=LAB[kind])
        ax.set_xscale("log")
        ax.set_title(f"({'abcde'[models.index(m)]}) {short[m]}", fontsize=12, loc="left", fontweight="bold")
        ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([150, 300, 700, 1500] if m == "arch" else [300, 600, 1200, 2500])
        lo_, hi_ = ax.get_ylim(); mid = 0.5 * (lo_ + hi_); ax.set_ylim(0.85 * mid, 1.15 * mid)
        ax.set_xlabel("Unknowns")
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Recovery error")
    h, l = axes[0, 0].get_legend_handles_labels()
    if shape[0] * shape[1] > len(models):
        axes.ravel()[-1].axis("off")
        axes.ravel()[-1].legend(h, l, loc="center", fontsize=13, frameon=False)
    else:
        axes[0, 0].legend(h, l, fontsize=10)
    fig.savefig(OUT / fname, dpi=170)
    plt.close(fig)
    print("converge", fname)


def adaptive_figure(model="blob"):
    A = pickle.load(open(R24 / "adaptive" / "adaptive.pkl", "rb"))
    rows = A["rows"]
    spec, sv = _survey(model)
    ex = np.asarray(sv.electrode_x, float)
    zm = float(sv.z_max)
    fp = footprint_polygon(sv)
    lo, hi = 15, 800
    fig = plt.figure(figsize=(13.5, 6.2), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.85])
    for k, (r, lab) in enumerate(((0, "Initial mesh"), (3, "After three refinements"))):
        ax = fig.add_subplot(gs[k, 0:2])
        d = A["models"][(model, "voronoi", r)]
        mpb = model_section_from_arrays(d["poly_verts"], d["cell_idx"], d["rho_cell"], ax=ax, electrodes=ex, depth_exaggeration=1.0,
                                        x_lim=(ex.min(), ex.max()), z_lim=zm, colorbar=False, clim=(lo, hi), title=None, clip_verts=fp, xlabel="")
        _outlines(ax, spec)
        ax.set_xlim(ex.min(), ex.max()); ax.set_ylim(-zm, 0)
        ax.set_title(f"({'ab'[k]}) Voronoi, {lab.lower()}: {d['n']} unknowns", loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("Depth (m)")
        if k == 1:
            ax.set_xlabel("Distance (m)")
    cb = fig.colorbar(mpb, ax=fig.axes[:2], location="right", fraction=0.02, pad=0.01)
    format_log_colorbar(cb); cb.set_label(r"$\rho$ ($\Omega\cdot$m)")
    ax = fig.add_subplot(gs[:, 2])
    for kind, mk in (("voronoi", "o"), ("triangle", "s")):
        uni = [(r["n"], r["err"]) for r in rows if r["model"] == model and r["kind"] == kind and r["round"] == -1]
        ad = []
        for rd in range(4):
            s = [r for r in rows if r["model"] == model and r["kind"] == kind and r["round"] == rd]
            ad.append((np.mean([r["n"] for r in s]), np.mean([r["err"] for r in s])))
        ax.scatter(*zip(*uni), marker=mk, s=18, color=COL[kind], alpha=0.35, label=f"{LAB[kind]}, uniform (each seed)")
        ax.plot(*zip(*ad), mk + "-", color=COL[kind], lw=2, label=f"{LAB[kind]}, adaptive (mean)")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([300, 600, 1500, 3000, 6000])
    ax.set_xlabel("Unknowns"); ax.set_ylabel("Recovery error (log$_{10}\\rho$ RMS)")
    ax.set_title("(c)", loc="left", fontweight="bold", fontsize=12); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(OUT / "fig_adaptive.png", dpi=170)
    plt.close(fig)
    print("adaptive")


def prior_figure(model="blob"):
    P = pickle.load(open(R24 / "prior" / "prior.pkl", "rb"))
    spec, sv = _survey(model)
    ex = np.asarray(sv.electrode_x, float)
    zm = float(sv.z_max)
    fp = footprint_polygon(sv)
    lo, hi = 15, 800
    combos = [("base", "rect1", "Rectangular, no prior"), ("weights", "rect1", "Rectangular, boundary weights"),
              ("base", "voronoi", "Voronoi, no prior"), ("aligned", "voronoi", "Voronoi, aligned cells and weights")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 3.7), layout="constrained")
    for ax, (var, kind, lab), L in zip(axes.ravel(), combos, "abcd"):
        d = P["models"][(model, var, kind)]
        mpb = model_section_from_arrays(d["poly_verts"], d["cell_idx"], d["rho_cell"], ax=ax, electrodes=ex, depth_exaggeration=1.0,
                                        x_lim=(ex.min(), ex.max()), z_lim=zm, colorbar=False, clim=(lo, hi), title=None, clip_verts=fp, xlabel="")
        _outlines(ax, spec)
        ax.set_xlim(ex.min(), ex.max()); ax.set_ylim(-zm, 0)
        ax.set_title(f"({L}) {lab}, {d['n']} unknowns", loc="left", fontsize=12, fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Depth (m)")
    for ax in axes[1]:
        ax.set_xlabel("Distance (m)")
    cb = fig.colorbar(mpb, ax=axes.ravel().tolist(), location="right", fraction=0.02, pad=0.01)
    format_log_colorbar(cb); cb.set_label(r"$\rho$ ($\Omega\cdot$m)")
    fig.savefig(OUT / "fig_prior.png", dpi=170)
    plt.close(fig)
    print("prior")


if __name__ == "__main__":
    which = sys.argv[1:] or ["models", "converge", "adaptive", "prior"]
    for w in which:
        if w == "converge":
            converge_figure()
            converge_figure(("step", "basin", "blob", "arch", "emb"), "fig_converge_supp.png", (2, 3), (11.5, 6.4))
        else:
            {"models": models_figure, "adaptive": adaptive_figure, "prior": prior_figure}[w]()
