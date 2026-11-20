"""Two Geant4 readings: a bulk wavelength shifter's mean photon count (WLSMEANNUMBERPHOTONS, a weight and a parameter
key) and time profile ("delta", Geant4's default, or "exponential", the material's wls_time_profile setting); a
unified ground dielectric_metal of sigma_alpha 0, whose lobe Geant4 reflects about the global normal
(UNIFIED_METAL_FLAT, DIELECTRIC_METAL_FLAT), with no sigma_alpha key."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from optics_toy import read_toy

from lumen.build import library as glue
from lumen.build import spec
from lumen.format import schema
from lumen.media import spectra
from lumen.rules import optics as rules
from lumen.surfaces import types as legacy
from lumen.surfaces.outcomes import ABSORB, BACKSCATTER, DETECT, DIFFUSE, GLOSSY, SPECULAR
from lumen.transport import medium as tmedium

F32 = jnp.float32
CTX = dict(lam=F32(400.0), wi=jnp.asarray([0.0, 0.6, -0.8], F32), n=jnp.asarray([0.0, 0.0, 1.0], F32), n1=F32(1.3),
           n2=F32(1.0))  # fmt: skip


@pytest.fixture(scope="module")
def df(tmp_path_factory):
    return read_toy(tmp_path_factory)


def _with(material, **props):
    return dataclasses.replace(material, properties={**material.properties, **props})


def _const(key, v):
    return schema.Property(key, np.asarray(v, np.float64))


def test_bulk_mean_photon_count_is_the_weight_and_a_key(df):
    sh = _with(df.materials["Shifter"], WLSMEANNUMBERPHOTONS=_const("WLSMEANNUMBERPHOTONS", 0.65))
    assert rules.medium(sh).params["wls"]["yield"] == 0.65
    src = glue._medium_sources(sh)
    assert src[("wls", "yield")] == glue.Source("materials/Shifter/WLSMEANNUMBERPHOTONS", "constant")
    # without the constant G4OpWLS emits exactly one photon: no key (OUT)
    assert glue._medium_sources(df.materials["Shifter"])[("wls", "yield")].column == "fixed"
    values, info = spec.detector_values(dataclasses.replace(df, materials={"Shifter": sh}))
    assert values["materials/Shifter/WLSMEANNUMBERPHOTONS"] == 0.65
    assert info["materials/Shifter/WLSMEANNUMBERPHOTONS"].range == (0.0, float("inf"))
    tab = schema.Property("WLSMEANNUMBERPHOTONS", np.asarray([1.0, 1.0]), np.asarray([2.0, 3.0]))
    with pytest.raises(rules.OpticsRefused, match="must be a constant"):
        rules.medium(_with(sh, WLSMEANNUMBERPHOTONS=tab))


def test_time_profile_delta_is_geant4s_default_and_exponential_is_declared(df):
    sh = _with(df.materials["Shifter"], WLSTIMECONSTANT=_const("WLSTIMECONSTANT", 1.2))
    e = rules.medium(sh)
    assert e.medium.channels[0].time_profile == "delta" and e.params["wls"]["tau"] == 1.2
    o = schema.OpticsModel("geant4", 1, {"wls_time_profile": "exponential"}, "recipe", "the run's setting")
    declared = dataclasses.replace(sh, optics=o)
    name, entry, _ = glue.geant4_medium(declared)
    assert name == "Shifter" and entry.medium.channels[0].time_profile == "exponential"
    with pytest.raises(ValueError, match="wls_time_profile"):
        schema.OpticsModel("geant4", 1, {"wls_time_profile": "gaussian"})


def test_reemit_delay_by_profile():
    """delta: the constant itself (G4WLSTimeGeneratorProfileDelta); exponential: -tau log u."""
    s = spectra.geant4_spectrum("s", np.array([2.0, 3.0]), np.array([1.0, 1.0]))
    rand, tau = jnp.asarray([0.3, 0.25], F32), F32(1.2)
    lam_d, d_delta = tmedium.reemit_draw(s, tau, rand, lambda x: x.arrays(), "delta")
    lam_e, d_exp = tmedium.reemit_draw(s, tau, rand, lambda x: x.arrays(), "exponential")
    assert float(d_delta) == pytest.approx(1.2) and float(d_exp) == pytest.approx(-1.2 * np.log(0.25), rel=1e-6)
    assert float(lam_d) == float(lam_e)  # the wavelength draw does not depend on the profile


def _surface(value, eff=None, model="unified"):
    props = {"REFLECTIVITY": _const("REFLECTIVITY", 0.9), "SPECULARSPIKECONSTANT": _const("SPECULARSPIKECONSTANT", 0.2),
             "SPECULARLOBECONSTANT": _const("SPECULARLOBECONSTANT", 0.3),
             "BACKSCATTERCONSTANT": _const("BACKSCATTERCONSTANT", 0.1)}  # fmt: skip
    if eff is not None:
        props["EFFICIENCY"] = _const("EFFICIENCY", eff)
    return schema.Surface("mesh", "dielectric_metal", model, "ground", value, props)


def test_sigma_alpha_zero_reflects_the_lobe_about_the_normal():
    e = rules.surface(_surface(0.0))
    assert e.surface_type is legacy.UNIFIED_METAL_FLAT and "sigma_alpha" not in e.params
    p = np.asarray(e.surface_type.probs({k: F32(v) for k, v in e.params.items()}, CTX))
    # R 0.9: spike 0.2 + lobe 0.3 specular, backscatter 0.1, Lambertian the rest 0.4; 0.1 absorbed
    assert np.allclose(p[[SPECULAR, BACKSCATTER, DIFFUSE, ABSORB]], [0.45, 0.09, 0.36, 0.1], atol=1e-6)
    assert p[GLOSSY] == 0 and p[DETECT] == 0
    # the same shares as UNIFIED_METAL with its lobe folded into the spike
    u = np.asarray(legacy.UNIFIED_METAL.probs({**{k: F32(v) for k, v in e.params.items()}, "sigma_alpha": F32(0.1)},
                                              CTX))  # fmt: skip
    assert np.allclose(p[SPECULAR], u[SPECULAR] + u[GLOSSY]) and np.allclose(p[DIFFUSE], u[DIFFUSE])
    assert rules.surface(_surface(0.1)).surface_type is legacy.UNIFIED_METAL
    assert rules.surface(_surface(1.0)).surface_type is legacy.UNIFIED_METAL  # G4OpticalSurface's default value
    with pytest.raises(rules.OpticsRefused, match=r"outside \[0.005, 1.0\] \(or exactly 0\)"):
        rules.surface(_surface(1.2))


def test_sigma_alpha_zero_on_a_sensor():
    e = rules.sensor(_surface(0.0, eff=0.5), (300.0, 600.0))
    assert e.surface_type is legacy.DIELECTRIC_METAL_FLAT and e.params["eff"] == 0.5
    p = np.asarray(e.surface_type.probs({k: F32(v) for k, v in e.params.items()}, CTX))
    assert np.allclose(p[[SPECULAR, BACKSCATTER, DIFFUSE, DETECT, ABSORB]], [0.45, 0.09, 0.36, 0.05, 0.05], atol=1e-6)


def test_no_facet_no_value_key():
    """sigma_alpha 0 and polish 1 sample no facet: the surface value is not a parameter (OUT)."""
    assert not rules.samples_facet(_surface(0.0)) and rules.samples_facet(_surface(0.2))
    assert not rules.samples_facet(_surface(1.0, model="glisur")) and rules.samples_facet(_surface(0.5, model="glisur"))
    flat = _surface(0.0)
    assert not spec.value_read(flat) and spec.value_read(_surface(0.2))
    e = rules.surface(flat)
    src = glue._surface_sources(flat, e.params)
    assert src[("p_lobe",)] == glue.Source("surfaces/mesh/SPECULARLOBECONSTANT", "constant")
    assert not any(k[0] == "sigma_alpha" for k in src)
