"""The viewer's scene: a GDML read as lumen reads it, with lumen's reading of every material, surface and volume.

    view = build("sk.gdml", detector="sk")      # or build("sk"): a registered detector, its GDML under LUMEN_DATA
    view.payload                                 # the JSON the page draws (``lumen-view/1``); view.errors

The GDML is imported and, with a recipe, the recipe is applied. Each material is read by its optics model: opaque, a
medium and its channels, or refused with lumen's message; each property is marked with what reads it (a medium
parameter and its key, a scintillation source, or nothing). Each surface gets its lumen surface type (or the sensor
reader's, on a sensor), its parameters and where it is attached. Every placed copy of every volume becomes an instance
with its world matrix. A step that fails is reported in ``errors`` and the rest is still built. Lengths m, energies
eV, wavelengths nm.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import time
import warnings
from pathlib import Path

import numpy as np

from lumen import _env
from lumen.format import schema
from lumen.gdml import importer, reader
from lumen.gdml.solids_schema import ATTRIBUTES
from lumen.units import HC
from lumen.viewer.mesh import Tessellator

SCHEMA = "lumen-view/1"
MAX_INSTANCES = 3_000_000  # placed copies drawn at most; deeper copies past it are left out, and reported
# The axis drawn vertical for a registered detector, when it is not z.
# next100: nexus e798170 places the shielding pedestal at -y (Next100Shielding.cc).
UP = {"next100": "y"}
SCINT_PREFIXES = ("SCINTILLATION", "FAST", "SLOW", "YIELDRATIO", "RESOLUTIONSCALE", "PROTON", "DEUTERON", "TRITON",
                  "ALPHA", "ION", "ELECTRON")  # fmt: skip
LENGTH_KEYS = importer.LENGTH_KEYS | {"MIEHG"}
TIME_KEYS = ("WLSTIMECONSTANT", "SCINTILLATIONTIMECONSTANT1", "SCINTILLATIONTIMECONSTANT2", "FASTTIMECONSTANT",
             "SLOWTIMECONSTANT")  # fmt: skip


@dataclasses.dataclass
class View:
    """A built scene: ``payload`` (JSON-ready) and the steps that failed (``errors``: [(stage, message)])."""

    payload: dict
    errors: list


def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype).tobytes()).decode("ascii")


def short(name):
    """Return a GDML name as Geant4 shows it (the '0x...' pointer suffix dropped)."""
    return reader.g4_strip(name) if name else name


def load_recipe(spec):
    """Return the recipe ``NAME`` of a Python file ``path/to/file.py:NAME`` (a recipe of your own, not registered)."""
    import importlib.util

    path, _, attr = str(spec).rpartition(":")
    if not path or not attr:
        raise ValueError(f"--recipe {spec!r}: give FILE.py:NAME")
    mod_spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    if mod_spec is None or mod_spec.loader is None:
        raise ValueError(f"--recipe {spec!r}: {path} is not a Python file")
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)
    return getattr(module, attr)


def resolve(source, detector=None, recipe=None):
    """Return (GDML path, recipe or None, recipe inputs, name) of a GDML path or a registered detector's name.

    ``recipe`` (``FILE.py:NAME``) reads a GDML file with a recipe of your own; ``detector`` with a registered one's.
    """
    from lumen import detectors, plugins

    detectors.register_builtins()
    p = Path(str(source))
    if recipe is not None:
        r = load_recipe(recipe)
        return p, r, {}, r.name
    name = detector or (None if p.suffix.lower() == ".gdml" or p.exists() else str(source))
    if name is None:
        return p, None, {}, p.stem
    pkg = plugins.detectors.get(name)
    data = _env.data_dir()
    if p.suffix.lower() == ".gdml" or p.exists():
        gdml = p
    elif data is not None:
        gdml = data / pkg.source_dataset
    else:
        raise FileNotFoundError(f"{name!r}: give its GDML, or set LUMEN_DATA (its dataset is {pkg.source_dataset!r})")
    inputs = {k: data / v for k, v in pkg.inputs.items()} if data is not None else {}
    return gdml, pkg.recipe, inputs, name


def _caught(fn, stage, errors, notes):
    """Run ``fn`` recording its warnings in ``notes``; return its result, or None with the error in ``errors``."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = fn()
        except Exception as err:
            errors.append((stage, f"{type(err).__name__}: {err}"))
            out = None
    notes += [dict(stage=stage, kind=type(x.message).__name__, text=str(x.message)) for x in w]
    return out


def _fallback(path, errors, notes):
    """Return the geometry of a GDML the importer refuses: solids and structure only, materials without optics."""
    doc = _caught(lambda: reader.read(path, warn=False), "read", errors, notes)
    if doc is None:
        return None
    sol, meshes, _ = importer.solids(doc)
    vols, pls = importer.structure(doc)
    mats = {}
    for n, m in doc.materials.items():
        comp = schema.Composition(m.density_g_cm3 or 0.0, (), m.z, m.molar_mass_g_mol)
        mats[n] = schema.Material(n, m.state, comp, m.temperature, m.pressure)
    world = str(doc.world)
    return schema.DetectorFile(name=Path(path).stem, world=world, detector_volume=world, wavelength_range=(0.0, 0.0),
                               solids=sol, volumes=vols, placements=pls, materials=mats, meshes=meshes,
                               report=tuple(doc.report))  # fmt: skip


def read(path, recipe, inputs, errors, notes):
    """Return (detector file, sensor set or None): the recipe applied when it can be; what failed is in errors."""
    keys = recipe.material_keys if recipe is not None else None
    df = _caught(lambda: importer.read_gdml(path, keys), "import", errors, notes)
    if df is None:
        return _fallback(path, errors, notes), None
    if recipe is None:
        return df, None
    res = _caught(lambda: recipe.apply(df, Path(path), inputs), "recipe", errors, notes)
    if res is None:
        return _with_detector(df, recipe, errors, notes), None
    return res.detector_file, res.sensor_sets.get("default")


def guess_detector(df, materials):
    """Return the volume lumen's ``auto:`` rule picks over every transparent material, for a GDML without a recipe.

    None when no material has a refractive index.
    """
    from lumen.build import lowering

    media = [m for m, info in materials.items() if info["status"] == "medium"]
    if not media:
        return None
    try:
        return lowering.find_detector(df, "auto:" + ",".join(media)).lv
    except Exception:
        return None


def _with_detector(df, recipe, errors, notes):
    from lumen.build import lowering

    dv = _caught(lambda: lowering.find_detector(df, recipe.detector_volume), "detector volume", errors, notes)
    if dv is None:
        return df
    return dataclasses.replace(df, detector_volume=dv.lv, wavelength_range=recipe.wavelength_range)


def instances(df, detector_lv, max_instances=MAX_INSTANCES):
    """Return every placed copy from the world, depth-first: lv index, parent, placement, depth, in-detector, matrix.

    The matrix is (n, 12): the 3x3 rotation-and-scale rows then the position (m), mapping the volume's frame to the
    world's.
    """
    from lumen.build.lowering import placement_pose

    lvs = list(df.volumes)
    index = {lv: i for i, lv in enumerate(lvs)}
    lv_of: list = []
    parent: list = []
    pv: list = []
    depth: list = []
    inside: list = []
    mats: list = []
    stack = [(df.world, np.eye(3), np.zeros(3), -1, -1, 0, df.world == detector_lv)]
    poses = {}
    cut = 0
    while stack:
        lv, lin, t, par, p, d, ins = stack.pop()
        if len(lv_of) >= max_instances:
            cut += 1
            continue
        g = len(lv_of)
        lv_of.append(index[lv])
        parent.append(par)
        pv.append(p)
        depth.append(d)
        inside.append(ins)
        mats.append(np.concatenate([lin.reshape(-1), t]))
        for k in reversed(df.volumes[lv].daughters):
            if k not in poses:
                poses[k] = placement_pose(df.placements[k])
            lc, tc = poses[k]
            child = df.placements[k].volume
            stack.append((child, lin @ lc, lin @ tc + t, g, k, d + 1, ins or child == detector_lv))
    return dict(lv=np.asarray(lv_of, np.int32), parent=np.asarray(parent, np.int32), pv=np.asarray(pv, np.int32),
                depth=np.asarray(depth, np.int32), inside=np.asarray(inside, bool),
                matrix=np.asarray(mats, np.float64).reshape(-1, 12), cut=cut, lvs=lvs)  # fmt: skip


def _unit(key):
    if key in LENGTH_KEYS:
        return "m"
    if key in TIME_KEYS or key.endswith("TIMECONSTANT") or "RISETIME" in key:
        return "ns"
    if key.endswith("YIELD") or key in ("SCINTILLATIONYIELD",):
        return "1/MeV"
    return ""


SPECTRA = (
    "WLSCOMPONENT",
    "WLSCOMPONENT2",
    "SCINTILLATIONCOMPONENT1",
    "SCINTILLATIONCOMPONENT2",
    "SCINTILLATIONCOMPONENT3",
    "FASTCOMPONENT",
    "SLOWCOMPONENT",
)  # emission spectra: never marked as not covering the wavelength range


def _property(p: schema.Property, reads, lam):
    """Return a property for the page: its table (energies eV, wavelengths nm) or constant, and what reads it."""
    out = dict(key=p.key, origin=p.origin, note=p.note, matrix=short(p.matrix), unit=_unit(p.key), reads=reads)
    if p.energy is None:
        out.update(kind="constant", value=float(np.asarray(p.value)))
        return out
    e, v = np.asarray(p.energy, float), np.asarray(p.value, float)
    out.update(kind="table", energy=e.tolist(), value=v.tolist(), n=len(e), min=float(v.min()), max=float(v.max()),
               lam=[HC / float(e[-1]), HC / float(e[0])])  # fmt: skip
    if lam and lam[1] > lam[0] and p.key not in SPECTRA:
        out["covers"] = bool(e[0] <= HC / lam[1] + 1e-12 and e[-1] >= HC / lam[0] - 1e-12)
    return out


def _source_reads(sources):
    """Return {property key: [(lumen parameter, column, note)]} of a library entry's sources."""
    reads: dict = {}
    for path, src in (sources or {}).items():
        if src.key:
            prop = src.key.rsplit("/", 1)[-1]
            reads.setdefault(prop, []).append(dict(param="/".join(path), column=src.column, note=src.note,
                                                   key=src.key))  # fmt: skip
    return reads


def _reemission(entry):
    """Return {channel: its re-emission delay profile} of a medium's re-emitting channels (G4OpWLS)."""
    if entry is None:
        return {}
    return {c.name: getattr(c, "time_profile", "exponential") for c in entry.medium.channels if c.reemit is not None}


def _channels(entry):
    if entry is None:
        return []
    med = entry.medium
    out = []
    if med.index is not None:
        out.append("refraction")
    if med.absorption is not None:
        out.append("absorption")
    out += [c.name for c in med.channels]
    return out


def material_info(df, rec, lam):
    """Return a material as lumen reads it: its optics model, status, channels and every property's reader."""
    from lumen.build import library
    from lumen.rules import optics

    errors: list = []
    notes: list = []
    got = _caught(lambda: library._material_medium(rec), "medium", errors, notes)
    target, entry, sources = got if got is not None else (None, None, None)
    if errors:
        status = "refused"
    elif entry is None:
        status = "bound"  # lumen:same_as: another material's medium
    else:
        status = "opaque" if entry.medium.opaque else "medium"
    live = set(entry.params) if entry is not None else set()
    reads = _source_reads({k: v for k, v in (sources or {}).items() if k and k[0] in live})
    scint = _caught(lambda: optics.scintillation(rec), "scintillation", [], [])
    if "wls" in live and "WLSCOMPONENT" in rec.properties:  # read by the channel itself, not as a parameter
        note = "the re-emission spectrum (G4OpWLS)"
        reads.setdefault("WLSCOMPONENT", []).append(dict(param="wls/spectrum", column="spectrum", note=note, key=""))
    for k in rec.properties:
        if k.startswith(SCINT_PREFIXES) and scint is not None:
            reads.setdefault(k, []).append(dict(param="source", column="scintillation", note="lumen.sources", key=""))
    derived = []
    if status == "medium" and "RAYLEIGH" not in rec.properties:
        ray = optics.geant4_rayleigh(rec)
        if ray is not None:
            derived.append(_property(ray, [dict(param="rayleigh/length", column="derived", note=ray.note, key="")],
                                     lam))  # fmt: skip
    o = rec.optics
    return dict(
        name=rec.name, short=short(rec.name), state=rec.state, density=rec.composition.density_g_cm3,
        temperature=rec.temperature, pressure=rec.pressure,
        composition=[dict(ref=short(c.ref), fraction=c.fraction, atoms=c.atoms) for c in rec.composition.components],
        optics=dict(model=o.name, origin=o.origin, note=o.note, settings={k: str(v) for k, v in o.settings.items()}),
        status=status, target=target, message=errors[0][1] if errors else "", warnings=[n["text"] for n in notes],
        channels=_channels(entry), reemission=_reemission(entry),
        properties=[_property(p, reads.get(k, []), lam) for k, p in sorted(rec.properties.items())],
        derived=derived, inert=sorted(rec.inert),
        scintillation=None if scint is None else dict(yield_per_MeV=scint["yield_per_MeV"], tau_fast=scint["tau_fast"],
                                                      tau_slow=scint["tau_slow"], f_fast=scint["f_fast"]),
    )  # fmt: skip


def surface_info(df, rec, sensor_lvs, lam, pv_lv):
    """Return a surface as lumen reads it: its Geant4 form, lumen's surface type (or sensor reader's) and params."""
    from lumen.build import library
    from lumen.rules import optics

    errors: list = []
    notes: list = []
    model, finish, kind = optics.surface_names(rec)
    attached = [k.volume for k in df.skins if k.surface == rec.name]
    borders = [b for b in df.borders if b.surface == rec.name]
    on_sensor = any(lv in sensor_lvs for lv in attached) or any(
        pv_lv.get(b.pv2) in sensor_lvs or pv_lv.get(b.pv1) in sensor_lvs for b in borders
    )
    entry = _caught(lambda: library.surface_entry(rec), "surface", errors, notes)
    status, stype = ("ok", entry.surface_type.name) if entry is not None else ("refused", None)
    sensor_type = None
    if on_sensor:
        lam_domain = tuple(lam) if lam and lam[1] > lam[0] else None
        sens = _caught(lambda: optics.sensor(rec, lam_domain), "sensor", [], [])
        if sens is not None:
            status, sensor_type = "sensor", sens.surface_type.name
            entry = entry or sens
    sources = library._surface_sources(rec, entry.params) if entry is not None else {}
    reads = _source_reads(sources)
    value_name = optics.value_key(rec) if hasattr(optics, "value_key") else "value"
    return dict(
        name=rec.name, short=short(rec.name), geant4=dict(type=kind, model=model, finish=finish, value=rec.value,
                                                          value_name=value_name),
        optics=dict(model=rec.optics.name, origin=rec.optics.origin, note=rec.optics.note), status=status,
        surface_type=stype, sensor_type=sensor_type, message=errors[0][1] if (errors and status != "sensor") else "",
        refused_as_surface=errors[0][1] if errors else "",
        params=[dict(param="/".join(p), key=s.key, column=s.column, note=s.note) for p, s in sources.items()],
        properties=[_property(p, reads.get(k, []), lam) for k, p in sorted(rec.properties.items())],
        inert=list(rec.inert), skins=attached, borders=[[b.pv1, b.pv2] for b in borders[:200]], n_borders=len(borders),
    )  # fmt: skip


def _default(pre, post, materials):
    """Return what lumen does where no surface applies."""
    a, b = materials.get(pre or ""), materials.get(post or "")
    if pre == post:
        return "pass", "no boundary: the same material on both sides"
    if b is None or b["status"] == "opaque":
        return "absorb", "absorbed: no RINDEX beyond (Geant4's NoRINDEX)"
    if b["status"] == "refused" or (a is not None and a["status"] == "refused"):
        return "refused", "a material lumen refuses"
    return "fresnel", "Fresnel reflection or refraction between the two indices"


def _side(found, pre, post, ctx):
    """Return one crossing (pre -> post materials) given the surface found by the border and skin rules, if any."""
    materials, surfaces = ctx
    via, name = found
    a = materials.get(pre or "")
    if pre is not None and (a is None or a["status"] == "opaque"):  # None: an unnamed neighbour, taken transparent
        return dict(via="material", surface=None, outcome="none", text="no photon travels on this side (no RINDEX)")
    s = surfaces.get(name) if name else None
    if s is not None:
        g4 = s["geant4"]
        if pre == post and g4["type"] == "dielectric_dielectric" and g4["finish"] in ("polished", "ground"):
            out, what = _default(pre, post, materials)
            text = f"skipped (the same material on both sides): {what}"
            return dict(via=via, surface=name, outcome=out, text=text)
        if s["status"] == "sensor":
            return dict(
                via=via, surface=name, outcome="sensor", text=f"read by the sensor's reader: {s['sensor_type']}"
            )
        if s["status"] == "refused":
            return dict(via=via, surface=name, outcome="refused", text="a surface lumen refuses")
        return dict(via=via, surface=name, outcome="surface", text=f"surface type {s['surface_type']}")
    out, what = _default(pre, post, materials)
    return dict(via="material", surface=None, outcome=out, text=what)


@dataclasses.dataclass(frozen=True)
class _Boundaries:
    """A detector file's surfaces by where Geant4 looks them up: border pairs, borders by placement, skins by volume."""

    border: dict
    by_pv: dict
    skin: dict
    pv_lv: dict

    @classmethod
    def of(cls, df):
        by_pv: dict = {}
        for b in df.borders:
            by_pv.setdefault(b.pv1, []).append(b)
            by_pv.setdefault(b.pv2, []).append(b)
        return cls({(b.pv1, b.pv2): b.surface for b in df.borders}, by_pv, {k.volume: k.surface for k in df.skins},
                   {p.name: p.volume for p in df.placements})  # fmt: skip

    def find(self, pair, lv, mother):
        """Return (via, surface) of a crossing between a volume and its mother: the border, else the skins."""
        if pair in self.border:
            return "border", self.border[pair]
        for who in (lv, mother):
            if who in self.skin:
                return f"skin of {short(who)}", self.skin[who]
        return None, None


def _rows(df, bd, copy, ctx, group):
    """Return the crossings of one volume's first copy ``copy`` = (lv, pv, mother, mother's pv)."""
    lv, pv, mother, mpv = copy
    mat, mmat = df.volumes[lv].material, df.volumes[mother].material
    enter = dict(_side(bd.find((mpv, pv), lv, mother), mmat, mat, ctx), dir="in", partner=short(mother))
    if group is not None and group.rule == "any_entry":
        enter.update(via="sensor", surface=None, outcome="sensor", text=f"counted on entry (group {group.name})")
    rows = [enter, dict(_side(bd.find((pv, mpv), lv, mother), mat, mmat, ctx), dir="out", partner=short(mother))]
    if lv in bd.skin:  # the volume's own skin, met when a touching volume other than its mother enters it
        row = _side((f"skin of {short(lv)}", bd.skin[lv]), None, mat, ctx)
        text = row["text"] + " (unless that volume has a skin or a border of its own)"
        rows.append(dict(row, dir="in", partner="a touching volume", text=text))
    for b in bd.by_pv.get(pv, []):
        if mpv in (b.pv1, b.pv2):
            continue  # with the mother: shown above
        other = b.pv1 if b.pv2 == pv else b.pv2
        pre, post = df.volumes[bd.pv_lv[b.pv1]].material, df.volumes[bd.pv_lv[b.pv2]].material
        row = _side(("border", b.surface), pre, post, ctx)
        rows.append(dict(row, dir="in" if b.pv2 == pv else "out", partner=short(bd.pv_lv.get(other, other))))
    return rows


def crossings(df, inst, materials, surfaces, sset):
    """Return {lv: [crossing, ...]}: lumen's boundary rule at each volume's first copy, every border included.

    Geant4's order: an any-entry sensor; the border surface of the ordered pair of placed volumes; the skins (the
    daughter's, then the mother's, both ways); the material default. The crossings with the mother come first (in,
    then out), then the volume's own skin met from a touching volume, then every border surface the copy takes part in,
    with its partner. Coincident faces are not shown.
    """
    bd = _Boundaries.of(df)
    first: dict = {}
    for g, li in enumerate(inst["lv"]):
        first.setdefault(int(li), g)
    groups = sset.groups if sset is not None else ()
    out = {}
    for li, g in first.items():
        par = int(inst["parent"][g])
        if par < 0:
            continue
        lv, mother = inst["lvs"][li], inst["lvs"][int(inst["lv"][par])]
        mpv = df.placements[int(inst["pv"][par])].name if inst["pv"][par] >= 0 else None
        copy = (lv, df.placements[int(inst["pv"][g])].name, mother, mpv)
        out[lv] = _rows(df, bd, copy, (materials, surfaces), next((x for x in groups if x.selects(lv)), None))
    return out


def _solid_info(s):
    dims = dict(ATTRIBUTES.get(s.type, ()))
    params = {}
    for k, v in s.params.items():
        dim = next((d for a, d in dims.items() if a.lower() == k), "U")
        params[k] = [float(v) * 1e3, "mm"] if dim == "L" else ([float(np.degrees(v)), "deg"] if dim == "A" else
                                                                [float(v), ""])  # fmt: skip
    out = dict(type=s.type, params=params, refs=list(s.refs))
    if s.planes:
        out["planes"] = [[float(x) * 1e3 for x in row] for row in s.planes]
    return out


def _sensor_groups(sset):
    if sset is None:
        return []
    return [dict(name=g.name, rule=g.rule, surface=g.surface, lv=g.lv, lv_pattern=g.lv_pattern, model=g.model,
                 defaults=g.defaults, settings={k: str(v) for k, v in g.settings.items()})
            for g in sset.groups]  # fmt: skip


def _group_of(sset, lv):
    if sset is None:
        return None
    return next((g.name for g in sset.groups if g.selects(lv)), None)


def build(source, detector=None, max_instances=MAX_INSTANCES, recipe=None, up=None) -> View:
    """Return the scene (``View``) of a GDML file, or of a registered detector by name.

    ``detector`` reads the GDML with a registered detector's recipe, ``recipe`` (``FILE.py:NAME``) with your own;
    ``up`` (x, y or z) is the axis the page first draws vertical (``UP``, else z).
    """
    t0 = time.time()
    errors: list = []
    notes: list = []
    path, recipe, inputs, name = resolve(source, detector, recipe)
    df, sset = read(path, recipe, inputs, errors, notes)
    if df is None:
        return View(dict(schema=SCHEMA, meta=dict(name=name, source=str(path)), errors=errors, notes=notes), errors)
    lam = list(df.wavelength_range) if df.wavelength_range and df.wavelength_range[1] > 0 else None
    payload: dict = dict(schema=SCHEMA)
    payload["materials"] = {m: material_info(df, r, lam) for m, r in df.materials.items()}
    det_lv = df.detector_volume if recipe is not None else guess_detector(df, payload["materials"])
    inst = instances(df, det_lv, max_instances)
    sensor_lvs = {lv for lv in df.volumes if _group_of(sset, lv)}
    pv_lv = {p.name: p.volume for p in df.placements}
    payload["surfaces"] = {s: surface_info(df, r, sensor_lvs, lam, pv_lv) for s, r in df.surfaces.items()}
    payload.update(_geometry(df, inst, sset, payload["materials"]))
    cross = crossings(df, inst, payload["materials"], payload["surfaces"], sset)
    for v in payload["volumes"]:
        v["crossings"] = cross.get(v["name"])
    payload["placements"] = [
        dict(name=p.name, short=short(p.name), volume=p.volume, copy=p.copy) for p in df.placements
    ]
    payload["sensor_groups"] = _sensor_groups(sset)
    payload["sensor_ids"] = len(sset.ids) if sset is not None and sset.ids else None
    payload["meta"] = _meta(df, path, name, recipe, lam, dict(inst=inst, seconds=time.time() - t0))
    payload["meta"]["detector_volume"] = det_lv
    payload["meta"]["detector_guessed"] = recipe is None and det_lv is not None
    payload["meta"]["up"] = up or UP.get(detector or name, "z")
    payload["report"] = list(df.report)
    payload["notes"] = notes
    payload["errors"] = [dict(stage=s, text=t) for s, t in errors]
    return View(payload, errors)


def _meta(df, path, name, recipe, lam, run):
    import lumen

    inst, seconds = run["inst"], run["seconds"]
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return dict(name=name, source=str(path), file=Path(path).name, sha256=sha, lumen=lumen.__version__,
                world=df.world, detector_volume=df.detector_volume if recipe is not None else None,
                recipe=getattr(recipe, "name", None), wavelength_range=lam, provenance=_jsonable(df.provenance),
                counts=dict(solids=len(df.solids), volumes=len(df.volumes), placements=len(df.placements),
                            materials=len(df.materials), surfaces=len(df.surfaces), instances=int(len(inst["lv"])),
                            skins=len(df.skins), borders=len(df.borders)),
                instances_cut=int(inst["cut"]), outside_surface=getattr(df, "outside_surface", None),
                seconds=round(seconds, 2))  # fmt: skip


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)


def _geometry(df, inst, sset, materials):
    """Return the meshes, solids, volumes and instances of the page."""
    lvs = inst["lvs"]
    counts = np.bincount(inst["lv"], minlength=len(lvs))
    tess = Tessellator(df, {lv: int(counts[i]) for i, lv in enumerate(lvs)})  # coarser faces for repeated solids
    meshes: list = []
    mesh_of: dict = {}
    volumes: list = []
    border_pvs: dict = {}
    for b in df.borders:
        for side in (b.pv1, b.pv2):
            border_pvs.setdefault(side, []).append(b.surface)
    pv_lv = {p.name: p.volume for p in df.placements}
    lv_borders: dict = {}
    for pvn, surfs in border_pvs.items():
        lv_borders.setdefault(pv_lv.get(pvn), set()).update(surfs)
    skins: dict = {}
    for k in df.skins:
        skins.setdefault(k.volume, []).append(k.surface)
    for i, lv in enumerate(lvs):
        v = df.volumes[lv]
        if v.solid and v.solid not in mesh_of and counts[i] > 0:
            m = tess.solid(v.solid)
            mesh_of[v.solid] = len(meshes)
            meshes.append(dict(solid=v.solid, v=_b64(m.vertices, np.float32), t=_b64(m.triangles, np.uint32),
                               nv=len(m.vertices), nt=len(m.triangles), exact=m.exact, note=m.note))  # fmt: skip
        sel = inst["lv"] == i
        mat = materials.get(v.material) if v.material else None
        volumes.append(dict(
            name=lv, short=short(lv), solid=v.solid, solid_info=_solid_info(df.solids[v.solid]) if v.solid else None,
            mesh=mesh_of.get(v.solid), material=v.material, assembly=v.assembly, instances=int(counts[i]),
            inside=int(np.count_nonzero(inst["inside"][sel])), depth=int(inst["depth"][sel].min()) if counts[i] else -1,
            daughters=len(v.daughters), skins=skins.get(lv, []), borders=sorted(lv_borders.get(lv, [])),
            sensor=_group_of(sset, lv), status=mat["status"] if mat else None,
            aux=[dict(type=a.type, value=a.value) for a in v.aux],
        ))  # fmt: skip
    solids = {n: dict(type=s.type, mesh=mesh_of.get(n)) for n, s in df.solids.items()}
    data = dict(lv=_b64(inst["lv"], np.int32), parent=_b64(inst["parent"], np.int32), pv=_b64(inst["pv"], np.int32),
                depth=_b64(inst["depth"], np.int32), inside=_b64(inst["inside"], np.uint8),
                matrix=_b64(inst["matrix"], np.float32), n=int(len(inst["lv"])))  # fmt: skip
    return dict(meshes=meshes, solids=solids, volumes=volumes, instances=data)
