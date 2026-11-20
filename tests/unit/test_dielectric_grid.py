"""nexus's dielectric grid through the photon step: a mesh as one slab of xenon that absorbs as the mesh's transparency.

nexus (``/Geometry/Next100/use_dielectric_grid``) replaces each hexagonal mesh by a slab of thickness t made of a copy
of the gas (its RINDEX) with ABSLENGTH -t / ln T0, T0 the mesh's transparency: index matched, nothing reflects, and a
photon crossing at angle theta from the normal is transmitted with exp(-t / (L cos theta)) = T0^(1 / cos theta). A
pencil beam crosses such a slab (T0 = 0.9, t = 0.1 mm, as NEXT-100's EL grids) between two gas volumes onto a
counting sensor.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lumen
from lumen import run as runner
from lumen.format.sensorsets import SensorGroup
from lumen.geometry import tracked
from lumen.recipe import Recipe
from lumen.transport import item, state
from lumen.transport import step as steps
from lumen.units import HC
from lumen.validate import photons

T0, THICK = 0.9, 1e-4  # m
L = -THICK / math.log(T0)
N = 1 << 14
GDML = """<?xml version="1.0" encoding="UTF-8"?>
<gdml>
<define>
  <matrix name="RG" coldim="2" values="1.0*eV 1.0007 12.0*eV 1.0007"/>
  <matrix name="ABS" coldim="2" values="1.0*eV {L}*m 12.0*eV {L}*m"/>
</define>
<materials>
  <element name="Xe" formula="Xe" Z="54"><atom value="131.29"/></element>
  <material name="GXe" state="gas"><D value="0.08"/><fraction n="1.0" ref="Xe"/>
    <property name="RINDEX" ref="RG"/></material>
  <material name="GridMat" state="gas"><D value="0.08"/><fraction n="1.0" ref="Xe"/>
    <property name="RINDEX" ref="RG"/><property name="ABSLENGTH" ref="ABS"/></material>
  <material name="Sens" state="gas"><D value="0.08"/><fraction n="1.0" ref="Xe"/>
    <property name="RINDEX" ref="RG"/></material>
</materials>
<solids>
  <box name="WorldBox" x="40" y="40" z="2" lunit="m"/>
  <box name="GridBox" x="40" y="40" z="{t}" lunit="m"/>
  <box name="SensBox" x="40" y="40" z="0.01" lunit="m"/>
</solids>
<structure>
  <volume name="volGrid"><materialref ref="GridMat"/><solidref ref="GridBox"/></volume>
  <volume name="volSens"><materialref ref="Sens"/><solidref ref="SensBox"/></volume>
  <volume name="volWorld"><materialref ref="GXe"/><solidref ref="WorldBox"/>
    <physvol name="GRID"><volumeref ref="volGrid"/></physvol>
    <physvol name="SENS"><volumeref ref="volSens"/><position name="ps" x="0" y="0" z="-0.5" unit="m"/></physvol>
  </volume>
</structure>
<setup name="Default" version="1.0"><world ref="volWorld"/></setup>
</gdml>
"""


@pytest.fixture(scope="module")
def det(tmp_path_factory):
    d = tmp_path_factory.mktemp("grid")
    (d / "grid.gdml").write_text(GDML.format(L=L, t=THICK))
    groups = (SensorGroup("sens", "any_entry", "lumen:counting", lv="volSens", settings={"efficiency": 1.0}),)
    return lumen.load(lumen.convert(d / "grid.gdml", Recipe("grid", "volWorld", (150.0, 650.0), groups),
                                    d / "grid.lumen"))  # fmt: skip


@pytest.mark.parametrize("theta", [0.0, 30.0, 60.0, 75.0])
def test_a_dielectric_grid_transmits_its_transparency_to_the_power_one_over_cos(det, theta):
    t = math.radians(theta)
    d = np.array([math.sin(t), 0.0, -math.cos(t)])
    z = jnp.zeros(N, jnp.float32)
    x0, d0 = (jnp.tile(jnp.asarray(v, jnp.float32), (N, 1)) for v in (-0.3 * d, d))
    rows = state.Emission(x0, d0, z, z + HC / 7.2, z + 1.0, jnp.zeros((N, 2, 3), jnp.float32), jnp.full(N, True))
    model = steps.Model(tracked.from_parts(det.parts), det.parts.optics, steps.StepConfig())
    tally = item.HistogramShape(len(det.parts.optics.sensor_pvs), (150.0, 650.0))
    ops = steps.operands(model, photons.packed(det), None, steps.Controls(10.0, float("inf"), 0.0), item.grid(tally))
    out = runner.run(item.photons(tally), model, ops, rows, jax.random.PRNGKey(0), max_vertices=100,
                     replicas=1).outputs  # fmt: skip
    # absorption is a weight (G4OpAbsorption in expectation): every photon carries exactly T0^(1 / cos theta)
    assert float(np.sum(out["Q"])) / N == pytest.approx(T0 ** (1.0 / math.cos(t)), rel=2e-5)
