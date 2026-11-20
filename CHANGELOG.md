# Changelog

Changes a user of lumen can see, newest first. Each entry says what changed, what an old file, key or script does
now, and what to do.

## Unreleased

Added:

- `lumen view`: a browser viewer of a GDML file as lumen reads it (`lumen.viewer`, the `view` extra for exact
  booleans). It draws every placed volume, coloured by how lumen reads its material or by its optical surface, and
  shows for each material and surface lumen's status (a medium and its channels, opaque, a sensor, or refused with
  lumen's message), which lumen parameter each property feeds, and a plot of any table against wavelength. It reads a
  registered detector with its recipe (`lumen view sk`), any GDML without one, or with a recipe of your own
  (`--recipe FILE.py:NAME`). See `docs/howto/viewer.md`.
- A bulk wavelength shifter's WLSMEANNUMBERPHOTONS is read (G4OpWLS's mean photon count, a weight on the re-emitted
  photon, exact in expectation) instead of refused; its key `materials/<M>/WLSMEANNUMBERPHOTONS` is a parameter.
- A bulk WLSTIMECONSTANT is read under Geant4's default time profile, "delta" (the photon is re-emitted exactly that
  much later), instead of refused; a recipe whose Geant4 run uses "exponential" sets the material's optics setting
  `wls_time_profile` (`OpticsModel("geant4", 1, {"wls_time_profile": "exponential"}, "recipe", ...)`).
- A unified, ground dielectric_metal with sigma_alpha exactly 0 is read (Geant4 then reflects the lobe about the
  normal: surface types `unified_metal_flat`, `dielectric_metal_flat` on a sensor) instead of refused. Such a
  surface, and a glisur ground one of polish 1, has no `sigma_alpha` / `polish` key: Geant4 never reads that value
  there. A calibration that set one of those keys is refused as an unknown key: drop the key.
- A glisur, ground dielectric_dielectric surface is read (surface type `glisur_dielectric`) instead of refused:
  Geant4's REFLECTIVITY / TRANSMITTANCE cut, then DielectricDielectric's facet path, a Fresnel reflection or
  refraction about each GLISUR facet with the photon's polarisation until the photon leaves the surface on either
  side. Its REFLECTIVITY, TRANSMITTANCE and `polish` are parameters. A unified ground dielectric_dielectric is still
  refused.
- A unified sigma_alpha up to 1 rad is read (was 0.5): G4OpticalSurface's default value, which a surface built
  without one (nexus's SiPM board mask) has.
- A border surface between two volumes that meet only across coincident faces (neither mother and daughter nor
  siblings: nexus's TPB coating inside the light tube against the active gas, and its SiPM board coating, inside the
  mask inside the board, against the gas) is read: it is the surface of that crossing, as in Geant4, instead of
  refused. Such a border that no coincident face joins is refused at load ("the volumes never meet").
- A polished or ground dielectric_dielectric surface into a medium without RINDEX absorbs the photon there, as
  Geant4's NoRINDEX does, instead of being refused.
- A `border_into` sensor volume whose own skin surface detects (EFFICIENCY > 0) is a sensor where Geant4 picks that
  skin (sensor binding entry "skin"), as with nexus's photocathodes, instead of refused.
- `lumen.detectors.nexus`, the NEXT-100 recipe (`next100`, from nexus's GDML at `LUMEN_DATA/gdml/next100.gdml`,
  written with nexus's dielectric grids, `/Geometry/Next100/use_dielectric_grid true`, as `refs/gdml/export.sh`
  does): it converts, loads and tracks, with nexus's sensor ids. A GDML with the meshes built hole by hole loads, but
  its 165,000 holes overflow the navigator's surface codes: write it again with the dielectric grids.

Fixed:

- A bulk wavelength shifter re-emitted at any energy of its WLSCOMPONENT spectrum, also above the absorbed photon's.
  G4OpWLS redraws the energy up to 100 times until it is at most the absorbed photon's and drops the photon if it
  never is; lumen now draws from the spectrum below that energy, with the weight multiplied by the probability that
  one of the 100 draws lands there. Light that a shifter absorbs again (TPB's own blue emission, which TPB absorbs
  below 440 nm in nexus's NEXT-100) is no longer shifted back to bluer wavelengths: NEXT-100's PMTs see 3 to 5 %
  more light. A shifter that only absorbs light bluer than all of its emission is unchanged bit for bit.

## 0.1.3 (2026-11-11)

- tallymap 0.6.0, the final release (rc1 plus four fixes in its CPU backend and process spawning), installed from
  its tag v0.6.0; lumen accepts tallymap 0.6.x. The photon loop's results on the GPU are unchanged bit for bit.

- The developer documentation moved to `docs/dev/` (entry page `docs/dev/README.md`); this file replaces
  `docs/dev/migration.md`.
- Removed, unused by lumen and undocumented: `lumen.readout.grid.window_end`; `lumen.readout.hits.window_mass`,
  `window_horizon`, `poisson_nll_dark` and `first_hit_terms(horizon=)`; `Navigator.query(counts=, max_rounds=)`;
  `lumen.rules.optics.key_report` and `rt_sensor`, `lumen.validate.oracle.oracle_first_visible` and `mismatch`,
  `lumen.build.scene.check_partition`, `lumen.surfaces.polar.jones`, `lumen._core.safe.safe_norm` and
  `Compiled.is_single_convex` (test helpers, now in `tests/unit/helpers.py`).
- The geometry lowering's `dissolve` and `keep` options are removed: no build used them. The geometry check
  (`python -m lumen.validate.geometry`) now checks the geometry as simulated instead of a dissolved one.

Changed:

- Every material without a refractive index shares one opaque medium, by a general rule; the SBND and MicroBooNE
  recipes no longer bind theirs. The light is unchanged bit for bit on the five shipped detectors and no parameter key
  changes. A detector file converted again from SBND's or MicroBooNE's GDML differs in `materials.json` (those
  materials name the `geant4` optics model, not `lumen:opaque`), so a calibration saved against an earlier conversion
  of them is refused as made for another detector file: convert again and recalibrate, or keep the earlier file.

Fixed:

- JAX's persistent compilation cache stayed empty: JAX writes only programs that took over 1 s to compile, and each of
  lumen's compiles faster than that (the photon loop's heavy part is tallymap's CUDA kernel, cached by tallymap). lumen
  now caches every program, unless `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` is set. On the toy tank a new
  process's first `simulate` goes from about 22 s to about 9 s.

- A surface type with a value fixed by its model kept only the outcomes it had with every other parameter at 0.5, so
  under Geant4's chained cuts a unified dielectric_metal surface with a fixed EFFICIENCY lost its backscatter and
  Lambertian outcomes (no shipped detector had such a surface).
- A source-position derivative on a very large detector (about 40,000 sensors) could run out of GPU memory in the
  clearance estimate about the source: its rays now run in batches sized to a memory budget
  (`lumen.geometry.clearance.BYTES`, 1 GiB). The clearance values are unchanged.

## 0.1.2 (2026-11-11)

New:

- **Extending lumen** (`docs/howto/extending.md`, `examples/extending.py`): `lumen.Param` declares a parameter that
  becomes a `Params` key; a source, a sensor response, a medium model or a surface model is one function registered
  by name, with its parameters as keys; `lumen.validate.check_gradient` checks any derivative against finite
  differences on the same random numbers.
- `simulate(controls=, diagnostics=)`: the step limits and the per-photon diagnostics that `run.run` already took.
- One-line install from git: `pip install "lumen[cuda] @ git+https://github.com/OmarAlterkait/lumen.git"`. tallymap
  is installed from its tag (v0.6.0rc1); the `cuda` extra is `jax[cuda12]`.
- `lumen.validate`, `lumen.extend`, `lumen.diagnostics` and `lumen.plugins` are reachable after a bare `import lumen`.

Changed:

- **`FromArrays(drawn=True)` and a derivative that moves its photons.** With `wrt=["source/position"]` such a list now
  raises `GradientNotAvailable` unless it carries its direction density's gradient, `dlogp=` (N, 3). Before, lumen
  assumed every drawn list was isotropic: right for an isotropic list, silently wrong for any other (a cos^8 LED's
  position derivative was 5 times too small). To do: for isotropic directions pass `dlogp=jnp.zeros((N, 3))`, which
  gives the 0.1.1 result; for any other density pass grad_d log p (`lumen.sources.grad_logpdf`), or write the source
  as a function (`@lumen.source`). Every other source, detector file, sensor set and calibration is unchanged.
- **`Recipe.note` removed.** Nothing read it. A recipe written with `note=...` fails with a `TypeError`: drop the
  argument. WCSim's recipes also lost their `cathode_model` decision (a check that could not fail) and the sensor
  group's `name_prefix` setting (it equalled the default). Converted detector files are unchanged except the sensor
  set's JSON, which no longer lists `name_prefix`.
- **A border that starts outside the detector volume is dropped**, with the same warning as a border wholly outside,
  instead of raising "no adjacent world instances": Geant4 consults a border only for a photon in its first volume,
  and lumen tracks no light outside. Every detector that converted before converts to the same file. The reverse
  case, the detector volume into such a neighbour, still raises.
- A source function draws all its samples at once, so photon i is the same whatever the chunk size.

## 0.1.1 (2026-11-11)

Changed:

- The importer reads only Geant4's optical material keys by default. Any other material key is kept inert, with one
  grouped `InertKeysWarning`; `Recipe(material_keys=)` still reads more.
- A user GROUPVEL is no longer refused. It is not read, with a warning: photons are timed by RINDEX's group index
  (see [known limits](docs/explanation/known-limits.md)).
- Repair warnings are grouped by repair and object, with a count, and given once per conversion. Every occurrence
  stays in the detector file's report.
- `lumen.load(calibration=None)` is the nominal calibration.
- `simulate` refuses a callable source (or an array of photons) whose position is differentiated, with
  `GradientNotAvailable`; before, the derivative came out about 140 times too small, silently.
- `simulate` rescales very small source weights (largest below 2^-6) by an exact power of 2 and scales the outputs
  back. A source normalised to one photon was rouletted at its first event and rounded by the GPU's fixed-point sums.
  Weights of order 1 run unchanged, bit for bit.

Documentation: the health keys (`docs/reference/errors.md#health`), lost light at a junction of touching volumes,
split PMT interiors and `lumen:complex_index`, a large-detector timing, `CITATION.cff`.

## 0.1.0 (2026-11-10)

The first public release. Files and keys from the pre-release tag `0.0.1` are still read, with these renames:

| 0.0.1 | 0.1 | What an old file or key does |
|---|---|---|
| `surfaces/<S>/value` | `surfaces/<S>/sigma_alpha` (unified, rad) or `surfaces/<S>/polish` (glisur) | refused, naming the new key |
| a counting group's setting `eff`, key `sensors/<group>/eff` | `efficiency`, `sensors/<group>/efficiency` | a sensor set with `eff` is refused, naming the new setting; convert again |
| `Point(x=)`, `source/x` | `Point(position=)`, `source/position` | rename in scripts; positional use is unchanged |
| `SensorGroup.model` naming the surface reader (`lumen:geant4`, `lumen:counting`) | `SensorGroup.surface`; `model` is only the sensor model (`lumen:pmt`) | a `lumen-sensors/1` file is read as `lumen-sensors/2` |
| `build_options` in `detector.json` | removed (every build rewrite is a rule) | an empty one is read; a non-empty one is refused |
| health key `late` | `retired` (a photon stopped past `t_end`); `late` / `early` is light outside the time grid | – |
