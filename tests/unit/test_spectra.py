"""Emission spectra: the wavelength sampler follows its spectrum, and Geant4's WLS / scintillation sampler is the
literal Geant4 inverse."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumen.media import spectra
from lumen.transport import medium as tmedium
from lumen.units import HC

F32 = jnp.float32
TPB = spectra.wavelength_spectrum("tpb_like", [380, 400, 420, 440, 470, 520], [0.0, 0.6, 1.0, 0.8, 0.3, 0.0])


def chi2_vs_spectrum(lam, r, n_bins=25):
    edges = np.linspace(r.wl[0], r.wl[-1], n_bins + 1)
    grid = np.linspace(r.wl[0], r.wl[-1], 20001)
    dens = r.density_at(grid)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    exp = len(lam) * np.diff(np.interp(edges, grid, cdf))
    obs = np.histogram(lam, edges)[0]
    ok = exp > 20
    return float(np.sum((obs[ok] - exp[ok]) ** 2 / exp[ok]) / (ok.sum() - 1))


def test_wavelength_sampler_follows_its_spectrum():
    u = jax.random.uniform(jax.random.PRNGKey(0), (400_000,), F32, 1e-7, 1 - 1e-7)
    lam = np.asarray(jax.vmap(TPB.sample)(u), np.float64)
    assert lam.min() >= 380 and lam.max() <= 520
    assert chi2_vs_spectrum(lam, TPB) < 1.6
    grid = np.linspace(381.0, 519.0, 50)
    back = np.asarray(jax.vmap(TPB.sample)(jnp.asarray(TPB.rand_of(grid), F32)), np.float64)
    np.testing.assert_allclose(back, grid, atol=2e-3)  # rand_of is the inverse of sample


def test_wavelength_spectrum_refuses_bad_tables():
    with pytest.raises(ValueError):
        spectra.wavelength_spectrum("bad", [400.0, 390.0], [1.0, 1.0])


def geant4_sampler(energy, intensity, u):
    """G4OpWLS::BuildPhysicsTable (trapezoid integral at the knots) and G4PhysicsVector::GetEnergy (linear inverse)."""
    cii = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(energy) * (intensity[1:] + intensity[:-1]))])
    v = u * cii[-1]
    b = np.clip(np.searchsorted(cii, v, side="left") - 1, 0, len(cii) - 2)
    d = cii[b + 1] - cii[b]
    return energy[b] + np.where(d > 0, (v - cii[b]) * (energy[b + 1] - energy[b]) / np.where(d > 0, d, 1), 0.0)


def chi2_two(a, b, edges):
    ha, hb = np.histogram(a, edges)[0].astype(float), np.histogram(b, edges)[0].astype(float)
    ka, kb = math.sqrt(hb.sum() / ha.sum()), math.sqrt(ha.sum() / hb.sum())
    ok = ha + hb > 0
    return float(np.sum((ka * ha[ok] - kb * hb[ok]) ** 2 / (ha[ok] + hb[ok])) / max(ok.sum() - 1, 1))


def test_geant4_spectrum_sampler_is_geant4s():
    energy = np.array([2.5, 2.6, 2.75, 2.9, 3.0, 3.1, 3.4])
    intensity = np.array([0.0, 0.3, 1.0, 0.7, 0.4, 0.2, 0.0])
    sp = spectra.geant4_spectrum("shifter", energy, intensity)
    u = np.asarray(jax.random.uniform(jax.random.PRNGKey(0), (400_000,), F32, 1e-7, 1 - 1e-7))
    E = HC / np.asarray(jax.vmap(sp.sample)(jnp.asarray(u)), np.float64)
    Eg = geant4_sampler(energy, intensity, np.random.default_rng(1).random(400_000))
    grid = np.linspace(2.5, 3.4, 9001)
    assert chi2_two(E, Eg, np.linspace(2.5, 3.4, 61)) < 1.6
    assert abs(np.trapezoid(sp.density_energy(grid), grid) - 1) < 2e-3
    assert E.min() >= 2.5 - 1e-4 and E.max() <= 3.4 + 1e-4


def g4opwls(energy, intensity, absorbed_ev, n, rng):
    """G4OpWLS::PostStepDoIt (Geant4 11.3.0): each photon's energy redrawn up to 100 times until it is at most the
    absorbed photon's; NaN for a photon dropped after 100 misses."""
    e = geant4_sampler(energy, intensity, rng.random(n))
    for _ in range(99):
        miss = e > absorbed_ev
        if not miss.any():
            break
        e[miss] = geant4_sampler(energy, intensity, rng.random(int(miss.sum())))
    return np.where(e > absorbed_ev, np.nan, e)


@pytest.mark.parametrize("absorbed_ev", [2.9, 2.62, 2.505])
def test_reemission_is_below_the_absorbed_energy_as_g4opwls(absorbed_ev):
    energy = np.array([2.5, 2.6, 2.75, 2.9, 3.0, 3.1, 3.4])
    intensity = np.array([0.1, 0.3, 1.0, 0.7, 0.4, 0.2, 0.0])
    sp = spectra.geant4_spectrum("shifter", energy, intensity)
    n = 400_000
    eg = g4opwls(energy, intensity, absorbed_ev, n, np.random.default_rng(2))
    kept = float(tmedium.wls_kept(sp, F32(HC / absorbed_ev), lambda s: s.arrays()))
    assert kept == pytest.approx(np.isfinite(eg).mean(), abs=4 * math.sqrt(kept * (1 - kept) / n) + 1e-6)
    u = jax.random.uniform(jax.random.PRNGKey(4), (n, 2), F32, 1e-7, 1 - 1e-7)
    lam, _ = jax.vmap(lambda r: tmedium.reemit_draw(sp, F32(0.0), r, lambda s: s.arrays(), "delta",
                                                    F32(HC / absorbed_ev)))(u)  # fmt: skip
    e = HC / np.asarray(lam, np.float64)
    assert e.max() <= absorbed_ev * (1 + 1e-6)
    edges = np.linspace(2.5, absorbed_ev, 41)
    cdf = sp.rand_of(HC / edges)
    for sample in (e, eg[np.isfinite(eg)]):
        assert chi2_exact(sample, edges, np.diff(cdf) / cdf[-1]) < 1.6


def chi2_exact(sample, edges, p):
    h = np.histogram(sample, edges)[0].astype(float)
    expected = p * sample.size
    return float(np.sum((h - expected) ** 2 / expected) / (len(p) - 1))


def test_reemission_above_the_spectrum_is_the_plain_draw():
    sp = spectra.geant4_spectrum("shifter", np.array([2.5, 2.9, 3.4]), np.array([0.2, 1.0, 0.0]))
    u = jax.random.uniform(jax.random.PRNGKey(5), (1000, 2), F32)
    plain = jax.vmap(lambda r: tmedium.reemit_draw(sp, F32(0.0), r, lambda s: s.arrays()))(u)[0]
    cut = jax.vmap(lambda r: tmedium.reemit_draw(sp, F32(0.0), r, lambda s: s.arrays(), "exponential",
                                                 F32(HC / 3.5)))(u)[0]  # fmt: skip
    assert bool(jnp.all(plain == cut))
    assert float(tmedium.wls_kept(sp, F32(HC / 3.5), lambda s: s.arrays())) == 1.0


def test_wavelength_spectrum_below_is_its_cdf():
    sp = spectra.wavelength_spectrum("w", np.array([400.0, 450.0, 520.0]), np.array([0.2, 1.0, 0.1]))
    lam = np.array([390.0, 400.0, 431.0, 480.0, 520.0, 600.0])
    below = np.asarray(jax.vmap(sp.below)(jnp.asarray(lam, F32)), np.float64)
    assert np.allclose(below, 1.0 - sp.cdf_at(lam), atol=2e-6)
    draws = np.asarray(jax.vmap(lambda r: sp.sample_below(r, F32(431.0)))(jnp.linspace(1e-6, 1 - 1e-6, 2001)))
    assert draws.min() >= 431.0 - 1e-3 and draws.max() <= 520.0 + 1e-3
