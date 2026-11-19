"""A browser viewer of a GDML file as lumen reads it: the geometry, and what lumen does with each material and surface.

    lumen view sk                       # or: python -m lumen.viewer sk
    view = lumen.viewer.build("next100.gdml")

``scene.build`` reads the GDML (and a registered detector's recipe) and gathers lumen's reading; ``mesh`` draws the
solids; ``server`` serves the page (three.js, LUCiD's viewer style) on 127.0.0.1. Display only: nothing here changes
how a detector is converted or simulated. The 3D view needs the ``view`` extra (``manifold3d``) for exact booleans.
"""

from lumen.viewer.scene import View, build

__all__ = ["View", "build"]
