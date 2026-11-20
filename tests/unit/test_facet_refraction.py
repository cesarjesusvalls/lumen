"""GLISUR facet refraction (Geant4's ground glisur dielectric_dielectric): the path's density, its sampler, its type.

- One facet event's density, reflection plus refraction, integrates to 1 over the sphere: float64 quadrature of the
  float32 density, for s and p states, into a denser and into a lighter medium (the two Jacobians).
- The sampled paths against Geant4 11.3.0 (tests/data/g4_boundary, written by refs/g4boundary): the share that ends
  beyond the surface, and the (dz, |phi|) histogram of the last direction, by a two-sample chi2.
- Reweighting: paths drawn at one polish (one index) and weighted by the live / reference path density reproduce the
  paths drawn at the other, the premise of every derivative through the facet path.
- The surface type reads Geant4's R + T cut; through the photon step a path ends on either side of the surface: the
  facet toy's rings against Geant4's.
"""

import json
import math
from pathlib import Path

import facet_toy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from lumen._core import rng
from lumen.surfaces import facets
from lumen.surfaces import types as legacy
from lumen.surfaces.outcomes import ABSORB, GLOSSY, STRAIGHT

F32 = jnp.float32
DATA = Path(__file__).resolve().parents[1] / "data" / "g4_boundary"
FM = facets.FACETS_MAX
NZ = jnp.asarray([0.0, 0.0, 1.0], F32)
P_MIN, Z_MAX = 1e-3, 4.0


def _incident(theta):
    return jnp.asarray([math.sin(theta), 0.0, -math.cos(theta)], F32)


def _params(r, n1, n2, er):
    return dict(sigma_alpha=F32(r), n1=F32(n1), n2=F32(n2), er=er, ei=jnp.zeros(3, F32))


def _sphere(nc=1200, nphi=480):
    u = (np.arange(nc) + 0.5) / nc
    ct = 1.0 - 2.0 * u
    phi = (np.arange(nphi) + 0.5) / nphi * 2.0 * math.pi
    CT, PH = np.meshgrid(ct, phi, indexing="ij")
    st = np.sqrt(1.0 - CT * CT)
    W = np.stack([st * np.cos(PH), st * np.sin(PH), CT], -1).reshape(-1, 3)
    return W, 4.0 * math.pi / W.shape[0]


@pytest.mark.parametrize(("r", "n1", "n2", "theta"), [(0.99, 1.0, 1.67, 30.0), (0.5, 1.67, 1.0, 20.0)])
@pytest.mark.parametrize("state", [(0.0, 1.0, 0.0), (1.0, 0.0, 1.0)], ids=["s", "p"])
def test_one_facet_event_density_integrates_to_one(r, n1, n2, theta, state):
    wi = _incident(math.radians(theta))
    er = jnp.asarray(state, F32)
    er = er - jnp.dot(er, wi) * wi
    p = _params(r, n1, n2, er / jnp.linalg.norm(er))
    W, dw = _sphere()
    total = 0.0
    for refr in (False, True):

        def lp(w, refr=refr):
            return facets.glisur_dielectric_scan(p, wi, NZ, jnp.zeros((FM, 3), F32).at[0].set(w), 1,
                                                 jnp.zeros(FM, bool).at[0].set(refr))[0]  # fmt: skip

        f = jax.jit(jax.vmap(lp))
        total += sum(float(np.sum(np.exp(np.asarray(f(jnp.asarray(W[i:i + 200_000], F32)), np.float64))))
                     for i in range(0, len(W), 200_000)) * dw  # fmt: skip
    assert abs(total - 1.0) < 2e-3, total


def sample(r, n1, n2, theta, n, seed=0, chunk=1 << 17):
    """Return the last directions (n, 3), the facets used and whether each path ended; random linear states."""
    wi = _incident(theta)
    a = jnp.asarray([0.0, 1.0, 0.0], F32)
    b = jnp.cross(wi, a)

    def one(key):
        u = rng.uniforms(key, 4)
        er = jnp.cos(2.0 * math.pi * u[3]) * a + jnp.sin(2.0 * math.pi * u[3]) * b
        W, k, _, ended = facets.glisur_dielectric_path(_params(r, n1, n2, er), wi, NZ, u[:3])
        return W[k - 1], k, ended

    f = jax.jit(jax.vmap(one))
    out = [f(jax.vmap(lambda j: rng.photon_key(jax.random.PRNGKey(seed), j))(jnp.arange(i, i + chunk)))
           for i in range(0, n, chunk)]  # fmt: skip
    return tuple(np.concatenate([np.asarray(o[c]) for o in out]) for c in range(3))


def histogram(d, edges_z, edges_phi):
    return np.histogram2d(d[:, 2], np.abs(np.arctan2(d[:, 1], d[:, 0])), bins=[edges_z, edges_phi])[0]


def two_sample_chi2(a, b, na, nb, min_count=10):
    """Return (chi2, ndf) of two histograms of na and nb entries, over the bins holding min_count entries in all."""
    m = (a + b) >= min_count
    d = a * math.sqrt(nb / na) - b * math.sqrt(na / nb)
    return float(np.sum(d[m] ** 2 / (a + b)[m])), int(m.sum()) - 1


CASES = json.loads((DATA / "MANIFEST.json").read_text())["cases"] if (DATA / "MANIFEST.json").exists() else {}


@pytest.mark.parametrize("case", sorted(c for c, v in CASES.items() if v["type"] == "dielectric_dielectric"))
def test_paths_against_geant4(case):
    c = CASES[case]
    with np.load(DATA / c["file"]) as g:
        hist, edges_z, edges_phi, launched = g["hist"], g["edges_z"], g["edges_phi"], int(g["launched"])
    n = 1 << 19
    d, k, ended = sample(1.0 - c["value"], c["n1"], c["n2"], math.radians(c["theta"]), n)
    assert ended.all() and k.max() < FM
    ours = histogram(d.astype(np.float64), edges_z, edges_phi)
    beyond = edges_z[1:] <= 0.0
    t_ours, t_ref = ours[beyond].sum() / n, hist[beyond].sum() / launched
    z = (t_ours - t_ref) / math.sqrt(t_ours * (1 - t_ours) / n + t_ref * (1 - t_ref) / launched)
    chi2, ndf = two_sample_chi2(ours, hist, n, launched)
    assert abs(z) < Z_MAX and stats.chi2.sf(chi2, ndf) > P_MIN, (case, z, chi2, ndf)


@pytest.mark.parametrize(("which", "ref", "live"), [("sigma_alpha", 0.99, 0.9), ("n2", 1.6, 1.67)])
def test_reweighting_to_another_polish_or_index(which, ref, live):
    """Paths drawn at ``ref`` and weighted exp(log p_live - log p_ref) against paths drawn at ``live``.

    The reference's support holds the live one: a GLISUR facet tilts at most asin(r), and a denser medium narrows the
    refracted cone.
    """
    theta, n = math.radians(40.0), 1 << 18
    wi = _incident(theta)
    base = dict(sigma_alpha=0.99, n1=1.0, n2=1.67)
    a = jnp.asarray([0.0, 1.0, 0.0], F32)
    b = jnp.cross(wi, a)

    def one(key, value):
        u = rng.uniforms(key, 4)
        er = jnp.cos(2.0 * math.pi * u[3]) * a + jnp.sin(2.0 * math.pi * u[3]) * b
        q = dict(base, **{which: value})
        p = _params(q["sigma_alpha"], q["n1"], q["n2"], er)
        W, k, C, _ = facets.glisur_dielectric_path(p, wi, NZ, u[:3])
        pl = dict(p, **{which: F32(live)})
        w = jnp.exp(facets.glisur_dielectric_scan(pl, wi, NZ, W, k, C)[0]
                    - facets.glisur_dielectric_scan(p, wi, NZ, W, k, C)[0])  # fmt: skip
        return W[k - 1, 2], w

    keys = jax.vmap(lambda j: rng.photon_key(jax.random.PRNGKey(3), j))(jnp.arange(n))
    dz_r, w = (np.asarray(x, np.float64) for x in jax.jit(jax.vmap(lambda k_: one(k_, ref)))(keys))
    keys = jax.vmap(lambda j: rng.photon_key(jax.random.PRNGKey(4), j))(jnp.arange(n))
    dz_l, _ = (np.asarray(x, np.float64) for x in jax.jit(jax.vmap(lambda k_: one(k_, live)))(keys))
    edges = np.linspace(-1.0, 1.0, 21)
    for sel_r, sel_l in ((dz_r < 0, dz_l < 0), (dz_r >= 0, dz_l >= 0)):
        mr, ml = (w * sel_r).mean(), sel_l.mean()
        z = (mr - ml) / math.sqrt((w * sel_r).var() / n + ml * (1 - ml) / n)
        assert abs(z) < Z_MAX, (which, z)
    hw = np.histogram(dz_r, edges, weights=w)[0] / n
    hw2 = np.histogram(dz_r, edges, weights=w * w)[0] / n
    hl = np.histogram(dz_l, edges)[0] / n
    var = (hw2 - hw * hw) / n + hl * (1 - hl) / n
    use = var > 0
    chi2 = float(np.sum((hw - hl)[use] ** 2 / var[use]))
    assert stats.chi2.sf(chi2, int(use.sum())) > P_MIN, (which, chi2, int(use.sum()))


def test_the_type_is_geant4s_cut_then_the_facet_path():
    ctx = dict(lam=F32(400.0), wi=_incident(0.3), n=NZ, n1=F32(1.0), n2=F32(1.67))
    p = {"R": F32(0.7), "T": F32(0.2), "polish": F32(0.01)}
    pr = np.asarray(legacy.GLISUR_DIELECTRIC.probs(p, ctx))
    np.testing.assert_allclose(pr[[GLOSSY, STRAIGHT, ABSORB]], [0.7, 0.2, 0.1], atol=1e-7)
    assert float(legacy.GLISUR_DIELECTRIC.lobe_params(p, ctx)["sigma_alpha"]) == pytest.approx(0.99)
    lobe = facets.FACET_LOBES[legacy.GLISUR_DIELECTRIC.glossy]
    assert lobe.refracts and not facets.FACET_LOBES[legacy.GLISUR_METAL.glossy].refracts


def test_index_matched_media_pass_straight_through():
    wi = _incident(0.6)
    p = _params(0.99, 1.4, 1.4, jnp.asarray([0.0, 1.0, 0.0], F32))
    W, k, C, ended = facets.glisur_dielectric_path(p, wi, NZ, jnp.asarray([0.3, 0.6, 0.9], F32))
    assert int(k) == 1 and bool(C[0]) and bool(ended)
    np.testing.assert_array_equal(np.asarray(W[0]), np.asarray(wi))
    assert float(facets.glisur_dielectric_scan(p, wi, NZ, W, k, C)[0]) == 0.0


def test_the_step_sends_each_path_to_its_side(tmp_path):
    """The facet toy through the photon step (tests/unit/facet_toy.py): every photon reaches a ring, and the light per
    5-degree ring (the direction after the surface; the far side past 90 degrees) agrees with Geant4's."""
    case = "glisur_dd_1.67_1.0_p0.01_th20"
    c = CASES[case]
    det = facet_toy.load(tmp_path, c["n1"], c["n2"], c["value"])
    n = 1 << 17
    q, out = facet_toy.rings(det, math.radians(c["theta"]), n)
    for k in ("lost", "abandoned", "escaped"):
        assert float(np.sum(out[k])) == 0.0, k
    ours = q.sum(0)
    assert ours.sum() == pytest.approx(n, rel=1e-6)
    with np.load(DATA / c["file"]) as g:
        ref, launched = g["rings"].astype(np.float64), int(g["launched"])
    chi2, ndf = two_sample_chi2(ours, ref, n, launched)
    assert stats.chi2.sf(chi2, ndf) > P_MIN, (chi2, ndf)
