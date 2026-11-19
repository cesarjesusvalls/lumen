# The Super-K and NEXT-100 GDML files

How the GDML files under `LUMEN_DATA/gdml/` that `lumen view sk` and `lumen view next100` read were written. Neither
WCSim nor nexus ships a GDML file: both build their geometry in C++, so each is built against Geant4 11.3.0 with GDML
support (the LCG_107 view, `x86_64-el8-gcc11-opt`, which also has ROOT, GSL and HDF5), and its geometry is written with
Geant4's own `G4GDMLParser` once it is constructed. No event is run.

| File | Program | Geometry | sha256 |
|---|---|---|---|
| `sk.gdml` (7.5 MB) | WCSim `e2dc7a48` (WCSim/WCSim) | `/WCSim/WCgeom SuperK`, PMTs placed one by one (`/WCSim/PMT/ReplicaPlacement false`: lumen reads no replica), tuning at WCSim's defaults | `3471937d…` |
| `next100.gdml` (36 MB) | nexus `e798170` (next-exp/nexus) | `Next100OpticalGeometry` at 15 bar, as `macros/NEXT100_S1_LT.*.mac` | `23255a97…` |

```bash
source <the LCG_107 view>/setup.sh                                       # Geant4 11.3.0 with GDML
git -C WCSim checkout e2dc7a48 && git -C WCSim apply refs/gdml/wcsim_e2dc7a48_geant4_11.patch
git -C nexus checkout e798170 && git -C nexus apply refs/gdml/nexus_e798170_gdml.patch
# build each with CMake (WCSim: -DWCSIM_HEPMC3_ENABLED=OFF), then:
WCSIM_SRC=... WCSIM_BUILD=... refs/gdml/export.sh sk OUT
NEXUS_SRC=... NEXUS_BUILD=... refs/gdml/export.sh next100 OUT
```

What the patches change (nothing in the geometry):

- **Both**: `main` creates a `G4GDMLParser`, whose messenger defines `/persistency/gdml/write`; CMake asks Geant4 for
  its `gdml` component.
- **WCSim** (written for Geant4 10.3): the Geant4 11 spellings it needs to build (`G4VisAttributes::GetInvisible()`,
  `G4SystemOfUnits.hh`, `G4NeutronCaptureProcess`, a G4String argument; the Gd neutron-capture final states and two
  removed scintillation setters left out: no event is run). Geant4 11 refuses two things Geant4 10 accepted, both in
  materials Super-K does not place: user property keys without `createNewKey` (the Geant4-10 scintillation keys of
  BGO, a coated cathode's imaginary index), given it, and property tables with descending energies (WCTE's glass,
  reversed; the WLS plastics, let through by an exception handler that reports each table). The GDML writer writes
  only the materials of placed volumes, so neither reaches `sk.gdml`.

`sk.gdml` imports into lumen with the `sk` recipe as lumen's own SK detector file does: 11,096 sensors, one repair
(`non_finite`: the GROUPVEL table Geant4 computes for a material without dispersion holds `inf`).
