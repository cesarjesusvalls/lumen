"""A browser viewer of a GDML file as lumen reads it: the geometry, and what lumen does with each material and surface.

    lumen view sk                       # or: python -m lumen.viewer sk
    view = lumen.viewer.build("next100.gdml")

Exact boolean solids need the ``view`` extra (``manifold3d``).
"""

from lumen.viewer.scene import View, build

__all__ = ["View", "build"]
