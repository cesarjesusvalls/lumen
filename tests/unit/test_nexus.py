"""lumen.detectors.nexus: NEXT-100 from nexus's GDML (LUMEN_DATA/gdml/next100.gdml; skipped without it) -- the sensors
nexus counts (60 PMTs, 3584 SiPMs, by their detecting skins) with nexus's sensor ids, the TPB and mesh readings, and
the surfaces lumen does not support yet, refused by name."""

import pytest

from lumen import _env
from lumen.build import library as glue
from lumen.build import lowering
from lumen.build import tree as vtree
from lumen.detectors import nexus
from lumen.format.sensorsets import SensorSet
from lumen.gdml import importer
from lumen.rules import optics
from lumen.surfaces import types as legacy


def _gdml():
    d = _env.data_dir()
    p = None if d is None else d / "gdml" / "next100.gdml"
    if p is None or not p.exists():
        pytest.skip("LUMEN_DATA/gdml/next100.gdml is not available")
    return p


@pytest.fixture(scope="module")
def result():
    with pytest.warns(UserWarning):  # GXe's ATTACHMENT and ELSPECTRUM are no optical keys: kept inert
        return importer.import_gdml(_gdml(), nexus.NEXT100)


def test_sensors_and_nexus_ids(result):
    df, cols = result.detector_file, result.sensor_columns
    assert df.detector_volume.startswith("GAS") and df.wavelength_range == nexus.WAVELENGTHS
    kinds = cols["kind"]
    assert len(result.sensor_sets["default"].ids) == len(kinds) == 3644
    pmt = sorted(i for i, k in zip(cols["nexus_id"], kinds, strict=True) if k == "pmt")
    sipm = sorted(i for i, k in zip(cols["nexus_id"], kinds, strict=True) if k == "sipm")
    assert pmt == list(range(60))  # SensorSD::FindSensorID: the EP_HOLE copy (depth 3)
    assert len(sipm) == 3584 and sipm[0] == 1000 and sipm[-1] == 56063  # 1000 x board + mask hole
    # the photocathodes are the sensitive volumes' own skins (G4OpBoundaryProcess with InvokeSD)
    loaded = lowering.lower(df, detector=df.detector_volume, sensors=result.sensor_sets["default"].is_sensor)
    t = vtree.tree_from_geometry(loaded)
    pvs = set(loaded.sensors.values())
    skins = glue.skin_sensor_lvs(df, t, pvs)
    assert sorted(s.split("0x")[0] for s in skins.values()) == ["PHOTOCATHODE", "SIPM_OPSURF"]
    assert isinstance(result.sensor_sets["default"], SensorSet)


def test_tpb_and_meshes(result):
    df = result.detector_file
    tpb = next(m for n, m in df.materials.items() if n.startswith("TPB"))
    e = optics.medium(tpb)
    wls = e.medium.channels[-1]
    assert wls.name == "wls" and wls.time_profile == "delta"  # nexus sets no time profile: Geant4's default
    assert e.params["wls"]["yield"] == 0.65 and e.params["wls"]["tau"] == 1.2
    for name, s in df.surfaces.items():
        if name.startswith(("GAS_CATHODE_MESH", "GAS_EL_MESH", "GAS_COPPER_PLATE")):
            assert optics.surface(s).surface_type is legacy.UNIFIED_METAL_FLAT, name  # sigma_alpha 0


def test_unsupported_surfaces_are_refused_by_name(result):
    refused = {}
    for name, s in result.detector_file.surfaces.items():
        if optics.nonzero(s.properties, "EFFICIENCY"):
            continue  # the photocathodes: read as sensors
        try:
            optics.surface(s)
        except optics.OpticsRefused as err:
            refused[name.split("0x")[0]] = str(err)
    assert sorted(refused) == ["SIPM_BOARD_MASK_OPSURF", "SIPM_BOARD_MASK_WLS_OPSURF", "SIPM_S13372_WLS_OPSURF",
                               "gas_tpb_teflon_surf", "tpb_sapphire_surf"]  # fmt: skip
    assert "sigma_alpha 1.0" in refused["SIPM_BOARD_MASK_OPSURF"]
    assert all("ground dielectric_dielectric" in v for k, v in refused.items() if k != "SIPM_BOARD_MASK_OPSURF")
