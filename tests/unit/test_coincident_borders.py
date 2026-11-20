"""Border surfaces across coincident faces, through the photon step: nexus's two patterns, as toys.

- tube: the gas (ACTIVE) and a thin coating (TPB) inside the opaque tube around it, the coating's inner face on the
  gas's outer one; borders ACTIVE -> TPB and TPB -> ACTIVE are mirrors, the end caps counting sensors. From a source
  at the centre every photon reaches a cap; without the borders the light that meets the side wall is absorbed in
  the coating's backing.
- board: a coating (WLS) at the top of an opaque mask, the mask at the top of an opaque board, the board in the gas
  (GASPV); the borders GASPV <-> WLS are mirrors, a sensor slab above. Light from a point between them, shot
  downwards, reaches the sensor in the share a ray trace gives.
"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import lumen
from lumen import run as runner
from lumen.format.sensorsets import SensorGroup
from lumen.geometry import tracked
from lumen.recipe import Recipe
from lumen.transport import item, state
from lumen.transport import step as steps
from lumen.units import HC
from lumen.validate import photons

N = 1 << 14

HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<gdml>
<define>
  <matrix name="RG" coldim="2" values="1.0*eV 1.0 12.0*eV 1.0"/>
  <matrix name="RT" coldim="2" values="1.0*eV 1.67 12.0*eV 1.67"/>
  <matrix name="ONE" coldim="2" values="1.0*eV 1.0 12.0*eV 1.0"/>
</define>
<materials>
  <element name="Hyd" formula="H" Z="1"><atom value="1.008"/></element>
  <material name="Gas" state="gas"><D value="0.001"/><fraction n="1.0" ref="Hyd"/>
    <property name="RINDEX" ref="RG"/></material>
  <material name="Sens" state="gas"><D value="0.001"/><fraction n="1.0" ref="Hyd"/>
    <property name="RINDEX" ref="RG"/></material>
  <material name="Tpb" state="solid"><D value="1.0"/><fraction n="1.0" ref="Hyd"/>
    <property name="RINDEX" ref="RT"/></material>
  <material name="Teflon" state="solid"><D value="2.0"/><fraction n="1.0" ref="Hyd"/></material>
</materials>
"""
TUBE = (
    HEAD
    + """<solids>
  <box name="WorldBox" x="3" y="3" z="3" lunit="m"/>
  <tube name="ActiveS" rmin="0" rmax="0.5" z="1.0" deltaphi="360" aunit="deg" lunit="m"/>
  <tube name="TubeS" rmin="0.5" rmax="0.6" z="1.0" deltaphi="360" aunit="deg" lunit="m"/>
  <tube name="TpbS" rmin="0.5" rmax="0.501" z="1.0" deltaphi="360" aunit="deg" lunit="m"/>
  <tube name="CapS" rmin="0" rmax="0.6" z="0.02" deltaphi="360" aunit="deg" lunit="m"/>
  <opticalsurface name="Mirror" model="glisur" finish="polished" type="dielectric_metal" value="1">
    <property name="REFLECTIVITY" ref="ONE"/></opticalsurface>
</solids>
<structure>
  <volume name="volActive"><materialref ref="Gas"/><solidref ref="ActiveS"/></volume>
  <volume name="volTpb"><materialref ref="Tpb"/><solidref ref="TpbS"/></volume>
  <volume name="volTube"><materialref ref="Teflon"/><solidref ref="TubeS"/>
    <physvol name="TPB"><volumeref ref="volTpb"/></physvol>
  </volume>
  <volume name="volCapUp"><materialref ref="Sens"/><solidref ref="CapS"/></volume>
  <volume name="volCapDn"><materialref ref="Sens"/><solidref ref="CapS"/></volume>
  <volume name="volWorld"><materialref ref="Gas"/><solidref ref="WorldBox"/>
    <physvol name="ACTIVE"><volumeref ref="volActive"/></physvol>
    <physvol name="TUBE"><volumeref ref="volTube"/></physvol>
    <physvol name="CAP_UP"><volumeref ref="volCapUp"/><position name="pu" x="0" y="0" z="0.51" unit="m"/></physvol>
    <physvol name="CAP_DN"><volumeref ref="volCapDn"/><position name="pd" x="0" y="0" z="-0.51" unit="m"/></physvol>
  </volume>
{borders}
</structure>
<setup name="Default" version="1.0"><world ref="volWorld"/></setup>
</gdml>
"""
)
TUBE_BORDERS = """  <bordersurface name="MirrorIn" surfaceproperty="Mirror">
    <physvolref ref="ACTIVE"/><physvolref ref="TPB"/></bordersurface>
  <bordersurface name="MirrorOut" surfaceproperty="Mirror">
    <physvolref ref="TPB"/><physvolref ref="ACTIVE"/></bordersurface>"""

BOARD = (
    HEAD
    + """<solids>
  <box name="WorldBox" x="3" y="3" z="3" lunit="m"/>
  <box name="GasBox" x="2.8" y="2.8" z="2.8" lunit="m"/>
  <box name="BoardS" x="1" y="1" z="0.1" lunit="m"/>
  <box name="MaskS" x="1" y="1" z="0.04" lunit="m"/>
  <box name="WlsS" x="1" y="1" z="0.001" lunit="m"/>
  <box name="SensS" x="2" y="2" z="0.02" lunit="m"/>
  <opticalsurface name="Mirror" model="glisur" finish="polished" type="dielectric_metal" value="1">
    <property name="REFLECTIVITY" ref="ONE"/></opticalsurface>
</solids>
<structure>
  <volume name="volWls"><materialref ref="Tpb"/><solidref ref="WlsS"/></volume>
  <volume name="volMask"><materialref ref="Teflon"/><solidref ref="MaskS"/>
    <physvol name="WLS"><volumeref ref="volWls"/><position name="pw" x="0" y="0" z="0.0195" unit="m"/></physvol>
  </volume>
  <volume name="volBoard"><materialref ref="Teflon"/><solidref ref="BoardS"/>
    <physvol name="MASK"><volumeref ref="volMask"/><position name="pm" x="0" y="0" z="0.03" unit="m"/></physvol>
  </volume>
  <volume name="volSens"><materialref ref="Sens"/><solidref ref="SensS"/></volume>
  <volume name="volGas"><materialref ref="Gas"/><solidref ref="GasBox"/>
    <physvol name="BOARD"><volumeref ref="volBoard"/></physvol>
    <physvol name="SENS"><volumeref ref="volSens"/><position name="ps" x="0" y="0" z="0.5" unit="m"/></physvol>
  </volume>
  <volume name="volWorld"><materialref ref="Gas"/><solidref ref="WorldBox"/>
    <physvol name="GASPV"><volumeref ref="volGas"/></physvol>
  </volume>
{borders}
</structure>
<setup name="Default" version="1.0"><world ref="volWorld"/></setup>
</gdml>
"""
)
BOARD_BORDERS = """  <bordersurface name="MirrorIn" surfaceproperty="Mirror">
    <physvolref ref="GASPV"/><physvolref ref="WLS"/></bordersurface>
  <bordersurface name="MirrorOut" surfaceproperty="Mirror">
    <physvolref ref="WLS"/><physvolref ref="GASPV"/></bordersurface>"""


def load(d, which, borders):
    if which == "tube":
        text = TUBE.format(borders=TUBE_BORDERS if borders else "")
        groups = tuple(SensorGroup(f"cap{c}", "any_entry", "lumen:counting", lv=f"volCap{c}",
                                   settings={"efficiency": 1.0}) for c in ("Up", "Dn"))  # fmt: skip
    else:
        text = BOARD.format(borders=BOARD_BORDERS if borders else "")
        groups = (SensorGroup("sens", "any_entry", "lumen:counting", lv="volSens", settings={"efficiency": 1.0}),)
    p = Path(d) / f"{which}_{int(borders)}.gdml"
    p.write_text(text)
    recipe = Recipe(f"{which}_toy", "volWorld" if which == "tube" else "volGas", (300.0, 600.0), groups)
    return lumen.load(lumen.convert(p, recipe, Path(d) / f"{which}_{int(borders)}.lumen"))


def emission(which, n, seed=0):
    rng = np.random.default_rng(seed)
    if which == "tube":
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        x = np.zeros((n, 3))
    else:
        ct, ph = rng.uniform(-1.0, -0.2, n), rng.uniform(0.0, 2.0 * math.pi, n)
        st = np.sqrt(1.0 - ct * ct)
        d = np.stack([st * np.cos(ph), st * np.sin(ph), ct], 1)
        x = np.tile([0.0, 0.0, 0.2], (n, 1))
    z = jnp.zeros(n, jnp.float32)
    return state.Emission(jnp.asarray(x, jnp.float32), jnp.asarray(d, jnp.float32), z, z + HC / 3.0, z + 1.0,
                          jnp.zeros((n, 2, 3), jnp.float32), jnp.full(n, True))  # fmt: skip


def detected(det, which):
    model = steps.Model(tracked.from_parts(det.parts), det.parts.optics, steps.StepConfig())
    tally = item.HistogramShape(len(det.parts.optics.sensor_pvs), (300.0, 600.0))
    ops = steps.operands(model, photons.packed(det), None, steps.Controls(10.0, float("inf"), 0.0), item.grid(tally))
    out = runner.run(item.photons(tally), model, ops, emission(which, N), jax.random.PRNGKey(0), max_vertices=4000,
                     replicas=1).outputs  # fmt: skip
    for k in ("lost", "abandoned", "escaped"):
        assert float(np.sum(out[k])) == 0.0, k
    return float(np.sum(out["Q"])) / N


def board_mirror_share(n=1 << 22):
    """Return the share of the board toy's light that the mirror at z = 0.05 sends to the sensor slab (ray trace)."""
    rng = np.random.default_rng(1)
    ct, ph = rng.uniform(-1.0, -0.2, n), rng.uniform(0.0, 2.0 * math.pi, n)
    st = np.sqrt(1.0 - ct * ct)
    dx, dy = st * np.cos(ph), st * np.sin(ph)
    s = (0.05 - 0.2) / ct
    x, y = s * dx, s * dy
    on = (np.abs(x) < 0.5) & (np.abs(y) < 0.5)
    s2 = (0.49 - 0.05) / -ct
    return float(np.mean(on & (np.abs(x + s2 * dx) < 1.0) & (np.abs(y + s2 * dy) < 1.0)))


def test_a_border_across_the_tube_wall_is_crossed(tmp_path):
    assert detected(load(tmp_path, "tube", True), "tube") > 1.0 - 5.0 / N
    assert detected(load(tmp_path, "tube", False), "tube") < 0.4


def test_a_border_across_two_opaque_volumes_is_crossed(tmp_path):
    share, f = detected(load(tmp_path, "board", True), "board"), board_mirror_share()
    assert abs(share - f) < 4.0 * math.sqrt(f * (1.0 - f) / N), (share, f)
    assert detected(load(tmp_path, "board", False), "board") < 0.1
