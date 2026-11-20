"""Geant4's navigation rules as lumen's navigator applies them: tolerances, surface identity, overlaps, stuck tracks.

The photon carries the placed volume it is in and the surface it sits on (its face and outward sign). Geant4's
G4Navigator decides a point's side of a surface within kCarTolerance by the direction of travel; lumen's tolerance is
the float32 resolution of the coordinates, the **band**. The rules, each with the Geant4 code it follows:

- **Tolerance.** Three sites, two forms (one form is on docs/dev/roadmap.md): ``band`` (the query and the CSG
  evaluator: ``8 eps (|x| + |t|) + 1e-9 m``), ``check_step_in`` (the overlap check's step in: ``K_ON`` bands at a
  point) and ``step_band`` (the step's push: ``8 eps max|x| + 1e-9 m``).
- **Surface identity.** The face of the item the photon sits on stays its face for ``K_ON`` bands along the ray. Another
  item's face within ``K_ON_OTHER`` bands of the start (perpendicular to it) is a face the photon sits on too, and
  within ``K_ON_INTERIOR`` bands for a start on no surface (after a bulk event, at a source): the direction decides
  which side it is on (G4Navigator's fEntering / fExiting), and a start on no surface is located beyond that face
  along its normal (as under **Exit**). A face that close along the ray but farther perpendicularly is looked for up
  to ``GRAZE`` times farther (1 / sin(0.25 deg) ~ 229). The far face of a volume thicker than ``K_ON_OTHER`` bands is
  never taken for the near one.
- **Entry tie.** Coincident daughter faces: the lowest-index daughter is entered (G4VoxelNavigation::ComputeStep).
- **Exit.** Leaving a volume, the photon is located in the highest-index sibling holding the point, then down into
  its daughters (G4Navigator::LocateGlobalPointAndSetup, LevelLocate). Coincident siblings (one solid, one pose):
  the lowest index fills the region; leaving it the photon is located in the highest-index other one. The point
  located is one step past the crossing along the ray, or along the face's normal on the side the ray goes to when
  the crossing is grazing (|d . n| below ``NORMAL_LOCATE_COS``), where a step along the ray barely leaves the face.
  A zero-length crossing found by the start's membership alone (a start a float32 step past a face) is the face
  first met back along the ray, so the boundary process gets its normal.
- **StepTooSmall.** The zero-length step out of a coincident partner just located in runs no boundary process: a
  straight continue event (G4OpBoundaryProcess's StepTooSmall status).
- **Facet rule.** A mesh facet is met only by a ray entering it (G4TriangularFacet::Intersect, DistanceToIn): a start
  behind its plane by at most ``FACET_BEHIND_BANDS`` bands is on it; the crossing's t comes from the facet's plane,
  ``FACET_SHORT_ULPS`` ulps short; a back facet met first means the photon is inside the opaque solid.
- **Stuck rule.** After ``ZERO_STEPS_BEFORE_PUSH`` zero-length steps in a row the photon is pushed
  ``PUSH_DISTANCE_TOLERANCES`` bands along its direction and located afresh; at ``ZERO_STEPS_BEFORE_KILL`` it is
  abandoned (G4Navigator's fActionThreshold_NoZeroSteps = 10, fAbandonThreshold_NoZeroSteps = 25; Geant4 pushes
  100 kCarTolerance).

Notes
-----
Geant4 sources: G4Navigator (LocateGlobalPointAndSetup, LevelLocate, the zero-step thresholds), G4VoxelNavigation
(ComputeStep), G4NormalNavigation, G4TriangularFacet, G4OpBoundaryProcess (StepTooSmall).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

FLOAT32_EPS = float(np.finfo(np.float32).eps)
BAND_ULPS = 8.0  # the band: this many float32 ulps of the coordinates
BAND_FLOOR = 1e-9  # m: the band's floor (a point at the origin)
K_ON = 4  # the face sat on stays the photon's face for this many bands along the ray
K_ON_OTHER = 2  # another item's face within this many bands of a start on a surface (perpendicular) is AT it
K_ON_INTERIOR = 0.5  # ... of a start on no surface
NORMAL_LOCATE_COS = 0.5  # a crossing with |d . n| below this is located beyond its face along the normal
GRAZE = 256  # ... looked for up to this many times farther along the ray (1 / sin(0.25 deg) ~ 229)
FACET_BEHIND_BANDS = 0.25  # a start behind a facet's plane by at most this many bands is on it
FACET_SHORT_ULPS = 2.0  # a mesh hit's t from its plane, this many ulps short of it
ZERO_STEPS_BEFORE_PUSH = 10  # G4Navigator fActionThreshold_NoZeroSteps
ZERO_STEPS_BEFORE_KILL = 25  # G4Navigator fAbandonThreshold_NoZeroSteps
PUSH_DISTANCE_TOLERANCES = 8  # the push, in bands (Geant4: 100 kCarTolerance)


def band(xa, t):
    """Return the float32 band of a crossing at ``t`` on a ray whose largest start coordinate is ``xa`` (m).

    Endpoints closer than this are one crossing; the photon's own surface is within it of the start.
    """
    return jnp.float32(BAND_ULPS * FLOAT32_EPS) * (xa + jnp.abs(t)) + jnp.float32(BAND_FLOOR)


def check_step_in(xe):
    """Return the overlap check's step in from an entry point ``xe`` (3,): ``K_ON`` bands of its coordinates (m)."""
    return jnp.float32(K_ON * BAND_ULPS * FLOAT32_EPS) * jnp.max(jnp.abs(xe)) + jnp.float32(K_ON * BAND_FLOOR)


def step_band(x):
    """Return the step's band at a point ``x`` (3,), the stuck rule's push unit (m)."""
    return jnp.float32(BAND_ULPS * FLOAT32_EPS) * jnp.max(jnp.abs(x)) + jnp.float32(BAND_FLOOR)


def entry_wins(t, item, best):
    """Return whether a daughter's event at ``t`` beats the best so far, by Geant4's entry-tie rule.

    The volume's own exit is visited first and sets the best; a daughter beats it on a tie, and between daughters at
    one t the lowest index wins. ``best`` is ``(t, item, leaf, face, zero, is_own)``.
    """
    return (t < best[0]) | ((t == best[0]) & (best[5] | (item < best[1])))


def level_locate_wins(item, found, excl, n):
    """Return whether daughter ``item`` replaces ``found`` as the volume holding a point: the highest index first.

    ``excl`` is the volume just left (never re-entered here), ``n`` the number of volumes (an item above is no unit).
    """
    return (item > found) & (item != excl) & (item < n)
