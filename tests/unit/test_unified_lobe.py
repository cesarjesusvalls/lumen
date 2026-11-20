"""The unified facet lobe over sigma_alpha in [SIGMA_MIN, SIGMA_MAX] (up to 1 rad, G4OpticalSurface's default value).

- Its tables (the facing share F and log zeta, float32 bicubic Hermite) against their float64 quadrature, between
  the nodes, over the whole range.
- One facet reflection's density integrates to 1 over the sphere at sigma_alpha 0.7 and 1.
- The facet path (Geant4's DielectricMetal lobe reflection, repeated while the photon points into the surface)
  against Geant4 11.3.0 (tests/data/g4_boundary): the (dz, |phi|) histogram of the last direction.
"""

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import lobe_contract
import numpy as np
import pytest
from scipy import stats

from lumen._core import rng
from lumen.rules import optics
from lumen.surfaces import facets

F32 = jnp.float32
DATA = Path(__file__).resolve().parents[1] / "data" / "g4_boundary"
NZ = jnp.asarray([0.0, 0.0, 1.0], F32)
P_MIN = 1e-3
CASES = json.loads((DATA / "MANIFEST.json").read_text())["cases"] if (DATA / "MANIFEST.json").exists() else {}


def test_the_tables_cover_the_range_and_match_their_quadrature():
    lo, hi = math.log(optics.SIGMA_MIN), math.log(optics.SIGMA_MAX)
    y = facets.TABLE_Y0 + facets.TABLE_DY * np.array([facets.TABLE_J_LO, facets.TABLE_J_HI])
    assert y[0] < lo and y[1] > hi  # both ends interior: no end node's one-sided derivative is used
    rs = np.random.default_rng(0)
    sig = np.exp(rs.uniform(lo, hi, 48))
    x = rs.uniform(0.0, 1.0, 48)
    worst_f = worst_z = 0.0
    for s, xx in zip(sig, x, strict=True):
        theta = math.atan2(1.0, s * math.tan(0.5 * math.pi * xx))  # x = (2/pi) atan2(cos theta, s sin theta)
        f32 = float(facets.front_fraction(F32(s), F32(math.cos(theta)), F32(math.sin(theta))))
        worst_f = max(worst_f, abs(f32 - float(facets.front_fraction_x(np.array([xx]), s)[0])))
        z64 = math.log(facets.tilt_norm(s) / (s * s))
        worst_z = max(worst_z, abs(float(facets.log_zeta(F32(s))) - z64))
    assert worst_f < 2e-4 and worst_z < 2e-4, (worst_f, worst_z)


@pytest.mark.parametrize("sigma", [0.7, 1.0])
@pytest.mark.parametrize("theta", [10.0, 60.0, 85.0])
def test_one_reflection_density_integrates_to_one(sigma, theta):
    """On a direction grid dense about the forward direction, where the density has its 1 / angle singularity."""
    t = math.radians(theta)
    wi = np.array([math.sin(t), 0.0, -math.cos(t)])
    G = lobe_contract.direction_grid(wi)
    lp = lobe_contract.eval_grid(lambda w: facets.unified_facet_logpdf(F32(sigma), jnp.asarray(wi, F32), NZ, w), G["W"])
    total = float(np.sum(np.exp(lp) * G["dw"]))
    assert abs(total - 1.0) < 1e-3, total


def two_sample_chi2(a, b, na, nb, min_count=10):
    m = (a + b) >= min_count
    d = a * math.sqrt(nb / na) - b * math.sqrt(na / nb)
    return float(np.sum(d[m] ** 2 / (a + b)[m])), int(m.sum()) - 1


@pytest.mark.parametrize("case", sorted(c for c, v in CASES.items() if v["model"] == "unified"))
def test_lobe_paths_against_geant4(case):
    c = CASES[case]
    with np.load(DATA / c["file"]) as g:
        hist, edges_z, edges_phi, launched = g["hist"], g["edges_z"], g["edges_phi"], int(g["launched"])
    t = math.radians(c["theta"])
    wi = jnp.asarray([math.sin(t), 0.0, -math.cos(t)], F32)
    p = {"sigma_alpha": F32(c["value"])}
    n, chunk = 1 << 19, 1 << 17

    def one(key):
        W, k = facets.UNIFIED.path(p, wi, NZ, rng.uniforms(key, 3))
        return W[k - 1], facets.UNIFIED.weight(p, wi, NZ, W[k - 1])

    f = jax.jit(jax.vmap(one))
    out = [f(jax.vmap(lambda j: rng.photon_key(jax.random.PRNGKey(1), j))(jnp.arange(i, i + chunk)))
           for i in range(0, n, chunk)]  # fmt: skip
    d = np.concatenate([np.asarray(o[0], np.float64) for o in out])
    w = np.concatenate([np.asarray(o[1], np.float64) for o in out])
    assert 1.0 - w.mean() < 1e-4  # paths still pointing into the surface after FACETS_MAX facets
    ours = np.histogram2d(d[:, 2], np.abs(np.arctan2(d[:, 1], d[:, 0])), bins=[edges_z, edges_phi], weights=w)[0]
    chi2, ndf = two_sample_chi2(ours, hist, n, launched)
    assert stats.chi2.sf(chi2, ndf) > P_MIN, (case, chi2, ndf)
