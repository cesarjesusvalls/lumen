#!/bin/bash
# Write the GDML of WCSim's Super-K and of nexus's NEXT-100 optical geometry, as lumen's sk and next100 read them.
#
#   export.sh sk      OUT_DIR      # needs WCSIM_SRC (WCSim e2dc7a48 with wcsim_e2dc7a48_geant4_11.patch), WCSIM_BUILD
#   export.sh next100 OUT_DIR      # needs NEXUS_SRC (nexus e798170 with nexus_e798170_gdml.patch), NEXUS_BUILD
#
# Both are built against Geant4 11.3.0 with GDML (the LCG_107 view: source its setup.sh first; it has ROOT, GSL and
# HDF5 too). Only the geometry is written: no event is run. The patches only let each program write GDML
# (G4GDMLParser's /persistency/gdml/write) and build under Geant4 11; WCSim's also lets Geant4 11 accept the
# property tables of materials Super-K does not use (descending energies, Geant4-10 keys). See README.md.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
what=${1:?sk or next100}; out=$(cd "${2:?the output directory}" && pwd)
case $what in
  sk)
    : "${WCSIM_SRC:?the patched WCSim source}" "${WCSIM_BUILD:?its build directory (src/WCSim, this_wcsim.sh)}"
    source "$WCSIM_BUILD/this_wcsim.sh"
    export WCSIM_BUILD_DIR=$WCSIM_SRC WCSIMDIR=$WCSIM_SRC   # macros/jobOptions.mac and data/ are read from here
    cd "$out"
    "$WCSIM_BUILD/src/WCSim" "$HERE/sk_gdml.mac" "$HERE/wcsim_tuning_defaults.mac" > wcsim_gdml.log 2>&1
    ;;
  next100)
    : "${NEXUS_SRC:?the patched nexus source}" "${NEXUS_BUILD:?its build directory (the nexus binary)}"
    sed "s|@OUT@|$out|g; s|@HERE@|$HERE|g" "$HERE/next100_gdml.init.mac" > "$out/next100_gdml.init.mac"
    sed "s|@OUT@|$out|g" "$HERE/next100_gdml.config.mac" > "$out/next100_gdml.config.mac"
    echo "/persistency/gdml/write $out/next100.gdml" > "$out/next100_gdml.write.mac"
    cd "$NEXUS_SRC"   # nexus reads macros/ and data/ relative to its source tree
    "$NEXUS_BUILD/nexus" -b -n 0 "$out/next100_gdml.init.mac" > "$out/nexus_gdml.log" 2>&1
    ;;
  *) echo "export.sh: sk or next100" >&2; exit 2 ;;
esac
ls -l "$out/$what.gdml"
sha256sum "$out/$what.gdml"
