# The shipped detectors

lumen ships recipes for six detectors. `lumen.load("<name>")` converts one from its GDML on first use, into your
cache (`LUMEN_CACHE`), and loads it; later loads reuse the converted file.

| Name | Detector | Sensors | PMT values (sensor model) |
|---|---|---|---|
| `sk` | Super-Kamiokande (WCSim's geometry) | 20-inch PMTs | `lumen:sk_r3600@2` |
| `wcte` | the Water Cherenkov Test Experiment (WCSim) | 3-inch PMTs | `lumen:wcte_r14374@2` |
| `annie` | ANNIE (ANNIE's WCSim) | its PMT types (LAPPDs are not sensors yet: [known limits](../explanation/known-limits.md)) | ANNIE's WCSim tables, per tube type |
| `sbnd` | SBND (sbndcode's full-simulation GDML) | PMTs and X-ARAPUCAs, counted on entry | none: counting sensors |
| `microboone` | MicroBooNE (LArSoft's public optics) | the optical-detector discs, counted on entry | none: counting sensors |
| `next100` | NEXT-100 (nexus's optical geometry), **converts but does not load yet** (below) | 60 PMTs and 3584 SiPMs, detected at their photocathode skins | none: the surfaces' EFFICIENCY, as in nexus |

NEXT-100 converts (`lumen.convert`, `lumen.gdml.import_gdml`), with nexus's sensor ids in its id map, but
`lumen.load("next100")` refuses it until lumen reads three more things nexus's geometry uses: ground
dielectric_dielectric surfaces (its TPB coatings), a unified lobe width of 1 rad (the SiPM board mask) and border
surfaces between volumes that meet only across a coincident face. Each refusal names the surface. The navigator
also cannot number the surfaces of the EL and cathode meshes' 165,000 hexagonal holes yet (its int32 surface codes
overflow).

## Where the GDML comes from

The GDML files and the inputs some recipes read (ANNIE's WCSim sources, SBND's channel map) are not in the
repository. lumen reads them from a data directory named by `LUMEN_DATA`, at fixed paths below it: `gdml/sk.gdml`,
`gdml/wcte.gdml`, `gdml/annie_v2.gdml`, `gdml/sbnd_v02_00_full_simulation.gdml`, `gdml/microboone.gdml`,
`gdml/next100.gdml` (nexus has no GDML of its own: write one from nexus with Geant4's `/persistency/gdml/write`,
`lumen.detectors.nexus` says how), and for SBND
`gdml/sbnd_pds_mapping.json`, for ANNIE `wcsim/annie_e88314b` and `wcsim/annie_tuning_parameters.mac`. Each comes
from its experiment's public software (WCSim, ANNIE's WCSim, sbndcode, LArSoft); a missing file raises an error
naming it.

```python
import lumen

det = lumen.load("microboone")              # converted into LUMEN_CACHE the first time
print(det)
print(len(det.params()), "parameters;", len(det.sensor_table()["name"]), "sensors")
```

## Using one

A shipped detector is a detector like any other: `det.params()` lists its parameters by GDML name, `lumen.simulate`
runs it, and your calibrations are saved under `LUMEN_DETECTORS` (`det.save_calibration`), never inside the shipped
file. To use your own PMT values, copy a shipped value set and edit it ([value sets](../reference/value-sets.md)).

What each recipe decides, and why, is in its module docstring: `lumen.detectors.wcsim`, `lumen.detectors.larsoft` and
`lumen.detectors.nexus`.
