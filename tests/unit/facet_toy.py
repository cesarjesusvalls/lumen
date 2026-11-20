"""The facet toy: a glisur ground dielectric_dielectric interface between medium A (the world, index n1, z > 0) and a
hemisphere B (index n2, z < 0) of radius R, as the skin of B. A pencil beam from A hits the interface at the centre;
36 counting rings of 5 degrees on R..R_OUT, one sensor group each, record where each photon leaves the centre, i.e.
its direction after the surface (Geant4: refs/g4boundary's ``rings``)."""

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

R, R_OUT, E_EV, RINGS = 1.0, 1.01, 3.0, 36
LAM = HC / E_EV
GDML = """<?xml version="1.0" encoding="UTF-8"?>
<gdml>
<define>
  <matrix name="RA" coldim="2" values="1.0*eV {n1} 12.0*eV {n1}"/>
  <matrix name="RB" coldim="2" values="1.0*eV {n2} 12.0*eV {n2}"/>
</define>
<materials>
  <element name="Hyd" formula="H" Z="1"><atom value="1.008"/></element>
  <material name="MatA" state="gas"><D value="0.001"/><fraction n="1.0" ref="Hyd"/>
    <property name="RINDEX" ref="RA"/></material>
  <material name="MatB" state="solid"><D value="1.0"/><fraction n="1.0" ref="Hyd"/>
    <property name="RINDEX" ref="RB"/></material>
  <material name="SensA" state="gas"><D value="0.001"/><fraction n="1.0" ref="Hyd"/>
    <property name="RINDEX" ref="RA"/></material>
  <material name="SensB" state="solid"><D value="1.0"/><fraction n="1.0" ref="Hyd"/>
    <property name="RINDEX" ref="RB"/></material>
</materials>
<solids>
  <orb name="WorldOrb" r="1.2" lunit="m"/>
  <sphere name="HemiB" rmin="0" rmax="{R}" startphi="0" deltaphi="360" starttheta="90" deltatheta="90" aunit="deg"
    lunit="m"/>
{solids}
  <opticalsurface name="Facets" model="glisur" finish="ground" type="dielectric_dielectric" value="{polish}"/>
</solids>
<structure>
  <volume name="volB"><materialref ref="MatB"/><solidref ref="HemiB"/></volume>
{volumes}
  <volume name="volWorld"><materialref ref="MatA"/><solidref ref="WorldOrb"/>
    <physvol name="pvB"><volumeref ref="volB"/></physvol>
{placements}
  </volume>
  <skinsurface name="FacetSkin" surfaceproperty="Facets"><volumeref ref="volB"/></skinsurface>
</structure>
<setup name="Default" version="1.0"><world ref="volWorld"/></setup>
</gdml>
"""


def write_gdml(path, n1, n2, polish):
    solids, volumes, placements = [], [], []
    d = 180.0 / RINGS
    for i in range(RINGS):
        mat = "SensA" if (i + 0.5) * d < 90.0 else "SensB"
        solids.append(f'  <sphere name="S{i}" rmin="{R}" rmax="{R_OUT}" startphi="0" deltaphi="360" '
                      f'starttheta="{i * d}" deltatheta="{d}" aunit="deg" lunit="m"/>')  # fmt: skip
        volumes.append(f'  <volume name="volRing{i}"><materialref ref="{mat}"/><solidref ref="S{i}"/></volume>')
        placements.append(f'    <physvol name="pvRing{i}"><volumeref ref="volRing{i}"/></physvol>')
    path.write_text(GDML.format(n1=n1, n2=n2, R=R, polish=polish, solids="\n".join(solids),
                                volumes="\n".join(volumes), placements="\n".join(placements)))  # fmt: skip
    return path


def load(directory, n1, n2, polish):
    """Return the loaded toy (one counting sensor group per ring: adjacent rings of one group would be one region)."""
    d = Path(directory)
    groups = tuple(SensorGroup(f"ring{i}", "any_entry", "lumen:counting", lv=f"volRing{i}",
                               settings={"efficiency": 1.0}) for i in range(RINGS))  # fmt: skip
    recipe = Recipe("facet_toy", "volWorld", (300.0, 600.0), groups)
    return lumen.load(lumen.convert(write_gdml(d / "facet.gdml", n1, n2, polish), recipe, d / "facet.lumen"))


def pencil(theta, n):
    """Return n unpolarised photons from A aimed at the centre at polar angle theta (rad) from -z."""
    dvec = np.array([math.sin(theta), 0.0, -math.cos(theta)])
    z = jnp.zeros(n, jnp.float32)
    return state.Emission(jnp.tile(jnp.asarray(-0.5 * dvec, jnp.float32), (n, 1)),
                          jnp.tile(jnp.asarray(dvec, jnp.float32), (n, 1)), z, z + LAM, z + 1.0,
                          jnp.zeros((n, 2, 3), jnp.float32), jnp.full(n, True))  # fmt: skip


def rings(det, theta, n, replicas=8, seed=3):
    """Return the charge per ring (replicas, RINGS) and the run's outputs."""
    model = steps.Model(tracked.from_parts(det.parts), det.parts.optics, steps.StepConfig())
    tally = item.HistogramShape(len(det.parts.optics.sensor_pvs), (300.0, 600.0))
    ops = steps.operands(model, photons.packed(det), None, steps.Controls(10.0, float("inf"), 0.0), item.grid(tally))
    out = runner.run(item.photons(tally), model, ops, pencil(theta, n), jax.random.PRNGKey(seed), max_vertices=2000,
                     replicas=replicas).outputs  # fmt: skip
    ring = np.array([int(pv.rsplit("pvRing", 1)[1].split("/")[0]) for pv in det.parts.optics.sensor_pvs])
    q = np.zeros((replicas, RINGS))
    q[:, ring] = np.asarray(out["Q"], np.float64).reshape(replicas, -1)
    return q, out
