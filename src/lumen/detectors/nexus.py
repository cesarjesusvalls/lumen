"""The NEXT-100 recipe, for the GDML of nexus's ``Next100OpticalGeometry`` (next-exp/nexus e798170).

The sensors are the PMT_PHOTOCATHODE and SIPM_S13372_SENSAREA volumes, detected where a photon is absorbed by their
own skin surface, with that surface's EFFICIENCY, as nexus counts them (its SensorSD on those volumes, with
``/process/optical/boundary/setInvokeSD``). The detector volume is GAS, the outermost xenon volume. The id map
gives each sensor's ``kind`` (pmt or sipm) and ``nexus_id``, the number nexus's ``SensorSD::FindSensorID`` gives it:
a PMT's copy number at depth 3; a SiPM's copy number at depth 4 times 1000 plus its copy number at depth 2.
"""

from __future__ import annotations

from lumen.format.sensorsets import SensorGroup
from lumen.recipe import Recipe

DETECTOR = "auto:gxe"
WAVELENGTHS = (150.0, 650.0)  # nm: xenon scintillation and TPB emission
PMT_LV, SIPM_LV = "PMT_PHOTOCATHODE", "SIPM_S13372_SENSAREA"
SENSORS = f"^({PMT_LV}|{SIPM_LV})(0x[0-9a-fA-F]+)?$"
# nexus e798170: Next100EnergyPlane.cc SetSensorDepth(3); Next100SiPMBoard.cc SetSensorDepth(2), SetMotherDepth(4),
# SetNamingOrder(1000)
PMT_DEPTH = 3
SIPM_DEPTH, SIPM_MOTHER_DEPTH, SIPM_ORDER = 2, 4, 1000


def _copy_at(copies: dict, pv: str, depth: int) -> int:
    """Return G4VTouchable::GetCopyNumber(depth) of a placed volume, from its path of physvol names.

    A tree PV is named by the physvols from the detector placement down, then its logical volume; depth 0 is the last
    physvol. A physvol placed several times in one mother carries '#k'.
    """
    return copies[pv.split("/")[:-1][-1 - depth].split("#")[0]]


def next100_columns(df, loaded, tree, inputs) -> dict:
    """Return the id map's per-sensor kind (pmt, sipm), nexus's sensor id and logical volume."""
    t = tree.tree_from_geometry(loaded)
    copies = {p.name: p.copy or 0 for p in df.placements}  # GDML omits a copy number of 0
    kind, ids, lvs = [], [], []
    for i in sorted(loaded.sensors):
        pv = loaded.sensors[i]
        lv = t.by_name[pv].lv
        if lv.startswith(PMT_LV):
            kind.append("pmt")
            ids.append(_copy_at(copies, pv, PMT_DEPTH))
        else:
            kind.append("sipm")
            ids.append(SIPM_ORDER * _copy_at(copies, pv, SIPM_MOTHER_DEPTH) + _copy_at(copies, pv, SIPM_DEPTH))
        lvs.append(lv)
    if len(set(ids)) != len(ids):
        raise ValueError("next100: nexus sensor ids are not unique")
    return {"kind": kind, "nexus_id": ids, "lv": lvs}


NEXT100 = Recipe(
    name="next100", detector_volume=DETECTOR, wavelength_range=WAVELENGTHS,
    groups=(SensorGroup("sensors", "border_into", "lumen:geant4", lv_pattern=SENSORS),),
    sensor_columns=next100_columns)  # fmt: skip
