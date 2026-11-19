"""The recipe for NEXT-100 from nexus's own geometry (next-exp/nexus e798170: Next100OpticalGeometry, 15 bar).

The GDML is nexus's geometry written by Geant4 11.3's G4GDMLParser (nexus has no GDML of its own: a G4GDMLParser
instance added to nexus.cc and ``/persistency/gdml/write`` after initialisation, as in macros/NEXT100_S1_LT.*.mac).
nexus semantics this recipe follows (e798170 source):

- The hit. nexus's SensorSD sits on the PMTs' PMT_PHOTOCATHODE and the SiPMs' SIPM_S13372_SENSAREA, and NexusApp
  turns on ``/process/optical/boundary/setInvokeSD``: a photon is counted when G4OpBoundaryProcess ends it with
  Detection at that volume's skin (PHOTOCATHODE and SIPM_OPSURF: dielectric_metal, unified, polished, REFLECTIVITY 0,
  the quantum efficiency as EFFICIENCY), the sensitive detector of the volume entered recording it. So the sensors
  are those volumes in one ``border_into`` group, whose own detecting skins are their surfaces
  (``lumen.build.library.skin_sensor_lvs``), read as stock Geant4 reads them (``lumen:geant4``): the efficiency
  applies during tracking, as in nexus, and no sensor model applies after it.
- The id map's columns: ``kind`` (pmt, sipm) and ``nexus_id``, SensorSD::FindSensorID's number: a PMT's copy number at
  depth 3 (its EP_HOLE), a SiPM's 1000 x the copy at depth 4 (its board) + the copy at depth 2 (its mask hole).
- The detector volume: GAS, the outermost xenon volume, which holds the field cage, both planes and the sensors.
- The wavelengths: xenon's scintillation (151-200 nm in nexus's GXe tables) to TPB's emission (381-602 nm).
- TPB re-emits 0.65 photons per absorption on average after 1.2 ns, under Geant4's default 'delta' time profile
  (nexus sets none): docs/dev/decisions.md, 20. The cathode and EL meshes and the energy plane's copper plate are
  unified ground metals of sigma_alpha 0 (75 % spike, the rest Lambertian): decision 21.

Not supported yet, so ``lumen.load`` refuses NEXT-100 until they are (each fails closed, naming the surface): the TPB
coatings' glisur ground dielectric_dielectric surfaces (polish 0.01: the light tube's, the sapphire windows', the SiPM
board mask's and the SiPMs') and the SiPM board mask's unified sigma_alpha of 1.0 rad.
"""

from __future__ import annotations

from lumen.format.sensorsets import SensorGroup
from lumen.recipe import Recipe

DETECTOR = "auto:gxe"  # GAS, the outermost volume of nexus's GXe
WAVELENGTHS = (150.0, 650.0)  # nm: xenon scintillation (151-200) and TPB emission (381-602)
PMT_LV, SIPM_LV = "PMT_PHOTOCATHODE", "SIPM_S13372_SENSAREA"  # the sensitive volumes (each name + the export's 0x...)
SENSORS = f"^({PMT_LV}|{SIPM_LV})(0x[0-9a-fA-F]+)?$"
PMT_DEPTH = 3  # Next100EnergyPlane: pmt_->SetSensorDepth(3)
SIPM_DEPTH, SIPM_MOTHER_DEPTH, SIPM_ORDER = 2, 4, 1000  # Next100SiPMBoard: SetSensorDepth(2), SetMotherDepth(4), 1000


def _copy_at(copies: dict, pv: str, depth: int) -> int:
    """Return G4VTouchable::GetCopyNumber(depth) of a placed volume, from its path of physvol names.

    A tree PV is named by the physvols from the detector placement down, then its logical volume; depth 0 is the last
    physvol. A physvol placed several times in one mother carries '#k'.
    """
    return copies[pv.split("/")[:-1][-1 - depth].split("#")[0]]


def next100_columns(df, loaded, tree, inputs) -> dict:
    """Return the id map's per-sensor kind (pmt, sipm), nexus's sensor id and logical volume."""
    t = tree.tree_from_geometry(loaded)
    copies = {p.name: p.copy or 0 for p in df.placements}  # G4GDMLWriteStructure writes no copynumber of 0
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


# nexus's export has no exporter defect Geant4 refuses (lumen.gdml.reader.REPAIRS): no repairs.
NEXT100 = Recipe(
    name="next100", detector_volume=DETECTOR, wavelength_range=WAVELENGTHS,
    groups=(SensorGroup("sensors", "border_into", "lumen:geant4", lv_pattern=SENSORS),),
    sensor_columns=next100_columns)  # fmt: skip
