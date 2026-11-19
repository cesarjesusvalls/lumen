# Look at a GDML file as lumen reads it

`lumen view` opens a GDML file in your browser: the geometry in 3D, and next to it what lumen does with every material
and every optical surface. Use it before you write a recipe, to see what lumen will refuse and why; after, to check
that the detector volume and the sensors are the ones you meant; and to look up a table (an absorption length, a
reflectivity) against wavelength without writing a script.

It needs the `view` extra (`pip install "lumen[view]"`, which adds `manifold3d` for the solids' booleans) and a browser
with WebGL. Nothing it shows changes how a detector is converted or simulated.

## Open a detector

```bash
lumen view sk                                    # a registered detector: its GDML under LUMEN_DATA, with its recipe
lumen view my_detector.gdml                      # any GDML file, without a recipe
lumen view my_detector.gdml --detector sk        # a GDML file with a registered detector's recipe
lumen view tank.gdml --recipe my_recipes.py:TANK # with a recipe of your own (the Recipe named TANK in that file)
```

The command reads the file the way `lumen.convert` does (the importer, then the recipe's decisions and the general
import rules), prints how long that took, and serves the page on `http://127.0.0.1:8765/` until you stop it with
Ctrl-C. `--open` opens a browser, `--port` picks another number. On a cluster node, run it there and forward 8765 from
your laptop (`ssh -L 8765:127.0.0.1:8765 <node>`), then open the address on the laptop.

Reading Super-K's GDML takes about 15 s and NEXT-100's about 20 s. To look again later without reading the file
again, write the scene once and serve it from the file:

```bash
lumen view sk --out sk.view.json
lumen view --scene sk.view.json
```

## What you see

**The 3D view** draws every placed copy of every volume. The colours (COLOR in the toolbar) say:

| COLOR | Each volume coloured by |
|---|---|
| LUMEN | how lumen reads its material: a transparent medium, a wavelength shifter, a scintillator, a sensor of the recipe, opaque (no RINDEX: lumen, as Geant4, stops a photon at its boundary), or refused |
| MATERIAL | its material |
| SURFACE | the optical surface on it (its skin, or a border it takes part in), red when lumen refuses that surface |
| DEPTH | its depth in the volume tree |

UP picks the axis drawn vertical, since a GDML file does not say which one is up. The default is z, as in LUCiD and
WCSim's tanks; `next100` uses y, as nexus does (its pedestal is at −y). `--up` sets it from the command line. The
triad in the corner shows the x (red), y (green) and z (blue) axes as you turn the view.

SHOW DETECTOR draws only the detector volume and what it contains, which is what lumen simulates; without a recipe the
viewer guesses it as lumen's `auto:` rule would (the largest volume of a transparent material) and marks it GUESSED in
the REPORT tab. CLIP cuts the view along x, y or z (the slider moves the plane, the arrows keep the other side), and
MEDIA sets how transparent the media are. Volumes placed more than 50,000 times (a mesh's holes, say) are not drawn
until you tick them in the VOLUMES list; the limit is in the settings (the gear).

**The sidebar** lists the volumes (as a tree), the materials, the optical surfaces, the sensor groups of the recipe,
and the REPORT: the file, the detector volume, the wavelength range, everything lumen refuses with its message, the
warnings of the import (the GDML repairs among them) and the import report.

**The right panel** shows what you click, in the 3D view or in a list:

- a volume: the path of the copy you clicked, its solid and its parameters (mm and degrees), its material, its skin and
  borders, and whether it is a sensor;
- a material: its optics model and status, the channels lumen gives it (refraction, absorption, Rayleigh, Mie,
  wavelength shifting), and each property with what reads it: which lumen parameter (its key, for a calibration), a
  scintillation source, or nothing, with the reason (GROUPVEL, for instance: lumen times photons by the group index
  of RINDEX). A table whose knots do not cover the recipe's wavelength range is marked CLAMPED. Keys no Geant4 optical
  process reads are listed apart, as kept inert;
- a surface: its Geant4 type, model, finish and value, the lumen surface type it becomes (or, on a sensor, the one the
  group's surface reader gives it), each lumen parameter and the property it comes from, and where it is attached.

Click a table in the properties to plot it against wavelength, with the recipe's range shaded and the flat
continuation lumen and Geant4 use outside the knots dashed. The settings switch the plot to a log scale or to photon
energy.

## From Python

```python
import lumen.viewer

view = lumen.viewer.build("examples/data/example_tank.gdml", recipe="examples/example_tank.py:RECIPE")
water = next(m for m in view.payload["materials"].values() if m["short"] == "Water")
print(water["status"], water["channels"])         # medium ['refraction', 'absorption', 'rayleigh']
```

`view.payload` is the JSON the page draws; `view.errors` lists the steps that failed (the page still shows the rest).
