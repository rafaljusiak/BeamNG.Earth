"""Poland's official 3D buildings (GUGiK's CityGML LoD2 models) in place of the map's extruded footprints.

BNE_PL_LOD2=on (off by default). Where the bundled index (data/pl_lod2_sheets.json.gz, tools/build_pl_lod2_index.py)
has a model sheet for the place, a road tile's buildings are GUGiK's: their real footprints, heights and roof shapes
(2015-2018, from 2012 airborne laser scanning; 219 of Poland's 380 powiats: ten whole voivodeships, Warsaw among them).

  * Data: one zip per powiat, https://opendata.geoportal.gov.pl/InneDane/Budynki3D/LOD2/<TERYT>_gml.zip, one CityGML
    file per 1:5,000 map sheet (~2.1 x 2.3 km, 1-80 MB of XML). The index says where each sheet's entry starts in its
    zip, so a sheet is one HTTP Range request (its compressed bytes, ~4 % of the XML), checked against the entry's size
    and CRC-32, parsed once as it streams in (only walls, roofs and the ground outline: lod2Solid repeats them) and
    kept as <cache>/pl_lod2/v<FORMAT>/<sheet>.npz. No sheet, a failed request, a changed zip: no models there (the
    Overture buildings stay; a failure marks the build incomplete, revisit.fault, so the tile is made again later).
  * Position: the models are in EPSG:2180 (PUWG 1992: GRS80 transverse Mercator, lon0 19 E, k0 0.9993, FE 500 km,
    FN -5,300 km), converted here (Krueger's series, millimetres) to latitude / longitude and the anchor's frame. Their
    heights (PL-KRON86-NH) are not used: each building keeps its height over its own ground outline (GroundSurface, one
    flat z per building) and stands on the game's ground (buildings.base_z, the walls reaching under it as others').
  * The models have no use, storeys or age (only a BDOT10k id and the roof type): plan() still chooses the look, from
    the attributes of the Overture footprints each model replaces (buildings.tile_buildings: a footprint mostly inside a
    model is dropped and lends it its class, floors and colours; footprints that no model covers stay, so buildings
    newer than 2012 are kept).

Licence: CC BY 4.0, GUGiK (credited in CREDITS.md and the in-game attribution line).
"""
import gzip
import json
import logging
import math
import os
import struct
import threading
import zlib
from collections import OrderedDict
from functools import lru_cache
from xml.etree import ElementTree as ET

import numpy as np

log = logging.getLogger("earthd.pl_lod2")

ON = os.environ.get("BNE_PL_LOD2", "off").lower() in ("on", "1", "true", "yes")
FORMAT = 1                  # the parsed sheets' layout (bump on a change: the cache's subdirectory)
INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pl_lod2_sheets.json.gz")
HOST = "opendata.geoportal.gov.pl"
MAX_TRIS = 4000             # a model with more triangles (a stadium's roof, a refinery) is left out: its footprint stays
MEM_SHEETS = 6              # parsed sheets kept in memory per process
MISS_TTL_S = 6 * 3600.0     # s a sheet that could not be had is not asked again (this process)
WALL, ROOF = 0, 1


# ------------------------------------------------------------------------------------------------ PUWG 1992 (EPSG:2180)

_A, _F = 6378137.0, 1 / 298.257222101       # GRS80
_N = _F / (2 - _F)
_AA = _A / (1 + _N) * (1 + _N ** 2 / 4 + _N ** 4 / 64)
_E = math.sqrt(2 * _F - _F * _F)
_ALPHA = (_N / 2 - 2 * _N ** 2 / 3 + 5 * _N ** 3 / 16, 13 * _N ** 2 / 48 - 3 * _N ** 3 / 5, 61 * _N ** 3 / 240)
_BETA = (_N / 2 - 2 * _N ** 2 / 3 + 37 * _N ** 3 / 96, _N ** 2 / 48 + _N ** 3 / 15, 17 * _N ** 3 / 480)
_DELTA = (2 * _N - 2 * _N ** 2 / 3 - 2 * _N ** 3, 7 * _N ** 2 / 3 - 8 * _N ** 3 / 5, 56 * _N ** 3 / 15)
_LON0, _K0, _FE, _FN = math.radians(19.0), 0.9993, 500000.0, -5300000.0


def to_puwg92(lat, lon):
    """(E, N) in metres of WGS84 / ETRF89 latitude and longitude (degrees; arrays or scalars)."""
    phi, lam = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64)) - _LON0
    t = np.sinh(np.arctanh(np.sin(phi)) - _E * np.arctanh(_E * np.sin(phi)))
    xi, eta = np.arctan2(t, np.cos(lam)), np.arctanh(np.sin(lam) / np.sqrt(1 + t * t))
    x, y = xi.copy(), eta.copy()
    for j, a in enumerate(_ALPHA, 1):
        x = x + a * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        y = y + a * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    return _FE + _K0 * _AA * y, _FN + _K0 * _AA * x


def from_puwg92(e, n):
    """(lat, lon) in degrees of PUWG 1992 (E, N) metres (arrays or scalars)."""
    xi = (np.asarray(n, np.float64) - _FN) / (_K0 * _AA)
    eta = (np.asarray(e, np.float64) - _FE) / (_K0 * _AA)
    x, y = xi.copy(), eta.copy()
    for j, b in enumerate(_BETA, 1):
        x = x - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        y = y - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(x) / np.cosh(y))
    lam = _LON0 + np.arctan2(np.sinh(y), np.cos(x))
    phi = chi.copy()
    for j, d in enumerate(_DELTA, 1):
        phi = phi + d * np.sin(2 * j * chi)
    return np.degrees(phi), np.degrees(lam)


# ------------------------------------------------------------------------------------------------ map sheets

# The national 1:5,000 division (as in the file names): the 1:1,000,000 IMW sheet (band letter: 4 degrees of latitude
# from the equator; zone: 6 degrees of longitude from 180 W), 144 sheets of 1:100,000 (12 x 12, numbered from the
# north-west), then quarters four times (A-D, a-d, 1-4, 1-4; north-west, north-east, south-west, south-east).
_D100 = (1 / 3, 0.5)                    # degrees (lat, lon) of a 1:100,000 sheet
_D5K = (_D100[0] / 16, _D100[1] / 16)   # ... of a 1:5,000 one: 1'15" x 1'52.5"


def sheet_of(lat, lon):
    """The name of the 1:5,000 sheet at (lat, lon), e.g. 'N-34-139-A-c-1-1' (northern hemisphere)."""
    band, zone = int(lat // 4), int((lon + 180) // 6) + 1
    top, left = (band + 1) * 4.0, (zone - 1) * 6.0 - 180.0
    row, col = int((top - lat) // _D100[0]), int((lon - left) // _D100[1])
    row, col = min(max(row, 0), 11), min(max(col, 0), 11)
    name = f"{chr(ord('A') + band)}-{zone}-{row * 12 + col + 1}"
    top, left = top - row * _D100[0], left + col * _D100[1]
    dlat, dlon = _D100
    for letters in ("ABCD", "abcd", "1234", "1234"):
        dlat, dlon = dlat / 2, dlon / 2
        r, c = int(top - lat >= dlat), int(lon - left >= dlon)
        name += "-" + letters[r * 2 + c]
        top, left = top - r * dlat, left + c * dlon
    return name


def sheet_bounds(name):
    """(south, west, north, east) in degrees of a 1:5,000 sheet's name. Pure."""
    p = name.split("-")
    band, zone, num = ord(p[0]) - ord("A"), int(p[1]), int(p[2]) - 1
    top, left = (band + 1) * 4.0 - (num // 12) * _D100[0], (zone - 1) * 6.0 - 180.0 + (num % 12) * _D100[1]
    dlat, dlon = _D100
    for q, letters in zip(p[3:7], ("ABCD", "abcd", "1234", "1234")):
        k = letters.index(q)
        dlat, dlon = dlat / 2, dlon / 2
        top, left = top - (k // 2) * dlat, left + (k % 2) * dlon
    return top - dlat, left, top, left + dlon


def sheets_in(south, west, north, east):
    """The names of the 1:5,000 sheets that a latitude / longitude box touches."""
    eps = 1e-9
    out = []
    lat = math.floor(south / _D5K[0]) * _D5K[0] + _D5K[0] / 2
    while lat - _D5K[0] / 2 < north - eps:
        lon = math.floor(west / _D5K[1]) * _D5K[1] + _D5K[1] / 2
        while lon - _D5K[1] / 2 < east - eps:
            out.append(sheet_of(lat, lon))
            lon += _D5K[1]
        lat += _D5K[0]
    return out


# ------------------------------------------------------------------------------------------------ the index

@lru_cache(maxsize=1)
def index():
    """The bundled index: {"base", "zips": {teryt: {...}}, "sheets": {name: [[teryt, offset, compressed, size, crc32,
    method], ...]}}; an empty one when it is missing."""
    try:
        with gzip.open(INDEX, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        log.warning("LoD2 index %s missing or unreadable: no Polish 3D buildings", INDEX)
        return {"base": "", "zips": {}, "sheets": {}}


def has_sheet(name):
    return name in index()["sheets"]


def covered(south, west, north, east):
    """Whether any model sheet touches the box (degrees). Cheap: a box outside Poland is answered without the index."""
    if north < 49.0 or south > 54.9 or east < 14.1 or west > 24.2:
        return False
    return any(has_sheet(s) for s in sheets_in(south, west, north, east))


# ------------------------------------------------------------------------------------------------ CityGML

def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _coords(ring_el):
    """(k, 3) of a gml:LinearRing (posList or pos), its closing point dropped."""
    vals = []
    for el in ring_el.iter():
        t = _local(el.tag)
        if t in ("posList", "pos", "coordinates") and el.text:
            vals.extend(el.text.replace(",", " ").split())
    a = np.asarray(vals, np.float64)
    if len(a) % 3:
        return None
    a = a.reshape(-1, 3)
    if len(a) > 1 and np.allclose(a[0], a[-1]):
        a = a[:-1]
    return a if len(a) >= 3 else None


def _area_xy(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _building(el):
    """One bldg:Building element: (id, roof type, ground z, footprint (k, 2), [(kind, [ring (k, 3), ...])]) or None.
    The polygons of its walls and roofs (its BuildingParts' too), each an outer ring and its holes; the footprint is
    its GroundSurface's largest outer ring."""
    bid, roof_type = el.get("{http://www.opengis.net/gml}id"), -1
    polys, grounds = [], []
    for sub in el.iter():
        t = _local(sub.tag)
        if t == "roofType" and sub.text:
            try:
                roof_type = int(sub.text.strip())
            except ValueError:
                pass
        elif t == "stringAttribute" and sub.get("name") == "buildingId":
            v = next((c.text for c in sub if _local(c.tag) == "value"), None)
            if v:
                bid = v.strip()
        elif t in ("WallSurface", "RoofSurface", "GroundSurface"):
            for poly in sub.iter():
                if _local(poly.tag) != "Polygon":
                    continue
                rings = []
                for part in poly:
                    pt = _local(part.tag)
                    if pt in ("exterior", "interior"):
                        r = _coords(part)
                        if r is not None:
                            if pt == "exterior":
                                rings.insert(0, r)
                            else:
                                rings.append(r)
                if not rings or len(rings[0]) < 3:
                    continue
                if t == "GroundSurface":
                    grounds.append(rings[0])
                else:
                    polys.append((WALL if t == "WallSurface" else ROOF, rings))
    if not grounds or not polys:
        return None
    fp = max(grounds, key=lambda r: abs(_area_xy(r)))
    zg = float(min(r[:, 2].min() for r in grounds))
    return bid, roof_type, zg, fp[:, :2], polys


def parse(chunks):
    """The buildings of a CityGML file read from an iterable of byte chunks (streamed: a sheet's 80 MB of XML is never
    whole in memory): a list of _building tuples."""
    parser = ET.XMLPullParser(events=("end",))
    out = []
    for chunk in chunks:
        parser.feed(chunk)
        for _ev, el in parser.read_events():
            if _local(el.tag) == "Building":
                b = _building(el)
                if b is not None:
                    out.append(b)
                el.clear()
    parser.close()
    for _ev, el in parser.read_events():
        if _local(el.tag) == "Building":
            b = _building(el)
            if b is not None:
                out.append(b)
    return out


def pack(blds):
    """The npz arrays of parsed buildings (several powiats' files of one sheet: a building in two of them once)."""
    seen, keep = set(), []
    for b in blds:
        if b[0] in seen:
            continue
        seen.add(b[0])
        keep.append(b)
    if not keep:
        return {"origin": np.zeros(2), "ids": np.array([], "U1"), "roof": np.zeros(0, np.int16),
                "zg": np.zeros(0, np.float64), "bbox": np.zeros((0, 4), np.float32), "fp_off": np.zeros(1, np.int32),
                "fp": np.zeros((0, 2), np.float32), "poly_off": np.zeros(1, np.int32),
                "poly_kind": np.zeros(0, np.uint8), "poly_ring_off": np.zeros(1, np.int32),
                "ring_off": np.zeros(1, np.int32), "verts": np.zeros((0, 3), np.float32), "ntri": np.zeros(0, np.int32)}
    origin = np.floor(np.min([b[3].min(0) for b in keep], 0))
    fp_off, fps, bbox = [0], [], []
    poly_off, poly_kind, poly_ring_off, ring_off, verts, ntri = [0], [], [0], [0], [], []
    nv = 0
    for bid, roof, zg, fp, polys in keep:
        f = fp - origin
        fps.append(f)
        fp_off.append(fp_off[-1] + len(f))
        lo, hi = f.min(0), f.max(0)
        for _k, rings in polys:
            for r in rings:
                lo, hi = np.minimum(lo, r[:, :2].min(0) - origin), np.maximum(hi, r[:, :2].max(0) - origin)
        bbox.append((lo[0], lo[1], hi[0], hi[1]))
        t = 0
        for kind, rings in polys:
            poly_kind.append(kind)
            for r in rings:
                v = np.c_[r[:, :2] - origin, r[:, 2] - zg]
                verts.append(v)
                nv += len(v)
                ring_off.append(nv)
            poly_ring_off.append(len(ring_off) - 1)
            t += sum(len(r) for r in rings) - 2 + 2 * (len(rings) - 1)
        poly_off.append(len(poly_kind))
        ntri.append(t)
    return {"origin": origin, "ids": np.array([b[0] for b in keep]), "roof": np.array([b[1] for b in keep], np.int16),
            "zg": np.array([b[2] for b in keep], np.float64), "bbox": np.array(bbox, np.float32),
            "fp_off": np.array(fp_off, np.int32), "fp": np.concatenate(fps).astype(np.float32),
            "poly_off": np.array(poly_off, np.int32), "poly_kind": np.array(poly_kind, np.uint8),
            "poly_ring_off": np.array(poly_ring_off, np.int32), "ring_off": np.array(ring_off, np.int32),
            "verts": np.concatenate(verts).astype(np.float32), "ntri": np.array(ntri, np.int32)}


# ------------------------------------------------------------------------------------------------ fetching

class Unavailable(IOError):
    """A sheet that could not be had now (offline, the host down, a changed zip)."""


def _inflate(data, method):
    """Chunks of a zip entry's bytes (8: deflate, 0: stored)."""
    if method == 0:
        for i in range(0, len(data), 1 << 20):
            yield data[i:i + (1 << 20)]
        return
    if method != 8:
        raise Unavailable(f"zip method {method}")
    d = zlib.decompressobj(-15)
    for i in range(0, len(data), 1 << 18):
        out = d.decompress(data[i:i + (1 << 18)])
        if out:
            yield out
    out = d.flush()
    if out:
        yield out


def _checked(chunks, size, crc):
    n, c = 0, 0
    for ch in chunks:
        n += len(ch)
        c = zlib.crc32(ch, c)
        yield ch
    if n != size or c != crc:
        raise Unavailable(f"entry is {n} bytes, CRC {c:08x}: the index says {size}, {crc:08x} (the zip changed?)")


LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")     # (signature 0x04034b50, ..., name length, extra length)
HEADER_SLACK = 512                                # bytes read past the fixed header for the name and extra field


def fetch_entry(fetcher, base, entry, sheet, kind="demand"):
    """The parsed buildings of one powiat's file of a sheet (entry: the index's row), fetched in one Range request."""
    import netfetch
    teryt, offset, csize, size, crc, method = entry[:6]
    url = f"{base}{teryt}_gml.zip"
    end = offset + LOCAL_HEADER.size + HEADER_SLACK + csize - 1
    try:
        r = fetcher.get(url, kind=kind, headers={"Range": f"bytes={offset}-{end}"})
    except netfetch.FetchError as e:
        raise Unavailable(f"{url}: {e}") from e
    body = r.body
    if r.status == 200:
        body = body[offset:end + 1]
    if len(body) < LOCAL_HEADER.size:
        raise Unavailable(f"{url}: {len(body)} bytes at {offset}")
    sig, *_rest, nlen, xlen = LOCAL_HEADER.unpack_from(body)
    name = body[LOCAL_HEADER.size:LOCAL_HEADER.size + nlen].decode("utf-8", "replace")
    if sig != 0x04034B50 or not name.endswith(f"{teryt}_{sheet}.gml"):
        raise Unavailable(f"{url}: no entry {teryt}_{sheet}.gml at {offset} (the zip changed?)")
    start = LOCAL_HEADER.size + nlen + xlen
    data = body[start:start + csize]
    if len(data) != csize:
        if start + csize > LOCAL_HEADER.size + HEADER_SLACK + csize:      # (a long extra field: the rest)
            r2 = fetcher.get(url, kind=kind, headers={"Range": f"bytes={offset + start}-{offset + start + csize - 1}"})
            data = r2.body if r2.status == 206 else r2.body[offset + start:offset + start + csize]
        if len(data) != csize:
            raise Unavailable(f"{url}: {len(data)} of {csize} bytes")
    try:
        return parse(_checked(_inflate(data, method), size, crc))
    except (ET.ParseError, zlib.error) as e:
        raise Unavailable(f"{url} {teryt}_{sheet}.gml: {e}") from e


class Sheet:
    """A parsed sheet's arrays (pack's) and their positions."""

    def __init__(self, name, arrays):
        self.name = name
        self.a = arrays
        self.origin = np.asarray(arrays["origin"], np.float64)
        self.n = len(arrays["ids"])

    def select(self, e0, n0, e1, n1):
        """Indices of the buildings whose box meets [e0, e1] x [n0, n1] (PUWG 1992 metres)."""
        b = self.a["bbox"].astype(np.float64) + np.r_[self.origin, self.origin]
        return np.flatnonzero((b[:, 2] >= e0) & (b[:, 0] <= e1) & (b[:, 3] >= n0) & (b[:, 1] <= n1))


class Source:
    """Sheets from the disk cache, else the host (one per process: source())."""

    def __init__(self, cache_dir, fetcher):
        self.dir = os.path.join(cache_dir, "pl_lod2", f"v{FORMAT}")
        self.fetcher = fetcher
        self._mem = OrderedDict()
        self._lock = threading.Lock()
        self._sheet_locks = {}
        self._missed = {}           # sheet -> time it could not be had

    def _path(self, name):
        return os.path.join(self.dir, f"{name}.npz")

    def sheet(self, name, kind="demand"):
        """The Sheet, or None where there are no models; raises Unavailable when it could not be had now."""
        idx = index()
        entries = idx["sheets"].get(name)
        if not entries:
            return None
        with self._lock:
            s = self._mem.get(name)
            if s is not None:
                self._mem.move_to_end(name)
                return s
            lk = self._sheet_locks.setdefault(name, threading.Lock())
        with lk:                    # (two tiles of one sheet: fetched and parsed once)
            with self._lock:
                s = self._mem.get(name)
            if s is None:
                s = self._load(name, entries, idx["base"], kind)
                with self._lock:
                    self._mem[name] = s
                    while len(self._mem) > MEM_SHEETS:
                        self._mem.popitem(last=False)
        return s

    def _load(self, name, entries, base, kind):
        import time
        p = self._path(name)
        try:
            with np.load(p, allow_pickle=False) as z:
                return Sheet(name, {k: z[k] for k in z.files})
        except (OSError, ValueError, KeyError):
            pass
        t = self._missed.get(name)
        if t is not None and time.monotonic() - t < MISS_TTL_S:
            raise Unavailable(f"{name}: not had earlier")
        t0 = time.perf_counter()
        try:
            blds = []
            for e in entries:
                blds.extend(fetch_entry(self.fetcher, base, e, name, kind))
        except Unavailable:
            self._missed[name] = time.monotonic()
            raise
        arrays = pack(blds)
        os.makedirs(self.dir, exist_ok=True)
        tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, p)
        log.info("LoD2 sheet %s: %d buildings from %d file(s) in %.1f s", name, len(arrays["ids"]), len(entries),
                 time.perf_counter() - t0)
        return Sheet(name, arrays)


_sources = {}
_sources_lock = threading.Lock()


def source(cache_dir, fetcher):
    """This process's Source for a cache directory."""
    with _sources_lock:
        s = _sources.get(cache_dir)
        if s is None:
            s = _sources[cache_dir] = Source(cache_dir, fetcher)
        return s


# ------------------------------------------------------------------------------------------------ a tile's models

class Model:
    """One building's model in the anchor's frame: ring (k, 2) its ground outline (x, y), verts (v, 3) (x, y, height
    over its ground), polys: [(kind, [(start, end) of each ring in verts: the outer first, then its holes])],
    roof_type (GUGiK's code: 1000 flat, 1030 gabled, 1130 combined, ...), id (BDOT10k), top: its highest point over
    its ground, eave: its main eaves' height (_eave), flat: the share of its roof's plan area sloping less than
    FLAT_DEG (GUGiK codes most flat-roofed blocks 1130, "combined")."""
    __slots__ = ("id", "ring", "verts", "polys", "roof_type", "top", "eave", "ntri", "flat")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def shifted(self, dx, dy):
        """A copy with x, y moved by (dx, dy) (into a tile's local frame)."""
        d = np.array([dx, dy])
        return Model(id=self.id, ring=self.ring + d, verts=np.c_[self.verts[:, :2] + d, self.verts[:, 2]],
                     polys=self.polys, roof_type=self.roof_type, top=self.top, eave=self.eave, ntri=self.ntri,
                     flat=self.flat)


FLAT_DEG = 10.0             # a roof polygon sloping less is flat


def _flat_share(verts, polys):
    """The share of the roofs' plan area sloping less than FLAT_DEG."""
    flat = tot = 0.0
    cz = math.cos(math.radians(FLAT_DEG))
    for k, rs in polys:
        if k != ROOF:
            continue
        s, e = rs[0]
        r = verts[s:e]
        n = np.zeros(3)
        for i in range(len(r)):
            a, b = r[i], r[(i + 1) % len(r)]
            n += ((a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1]))
        ln = float(np.linalg.norm(n))
        if ln <= 0:
            continue
        a = abs(n[2]) / 2           # (Newell's z: twice the plan area)
        tot += a
        flat += a if abs(n[2]) / ln >= cz else 0.0
    return flat / tot if tot > 0 else 1.0


def _eave(verts, polys, top):
    """The height of a model's main eaves: the roofs' lowest points, the median over their plan area (a low annex
    beside a tall block is not the block's eave)."""
    zs, ws = [], []
    for k, rs in polys:
        if k != ROOF:
            continue
        s, e = rs[0]
        r = verts[s:e]
        zs.append(float(r[:, 2].min()))
        ws.append(abs(_area_xy(r)))
    if not zs:
        return top
    o = np.argsort(zs)
    w = np.cumsum(np.asarray(ws)[o])
    if w[-1] <= 0:
        return float(min(zs))
    return float(np.asarray(zs)[o][np.searchsorted(w, w[-1] / 2)])


def _models(sheet, rows, frame):
    a = sheet.a
    out = []
    if not len(rows):
        return out
    # every vertex and outline point of the chosen buildings converted at once
    vs, fs, spans = [], [], []
    for i in rows:
        p0, p1 = a["poly_off"][i], a["poly_off"][i + 1]
        r0, r1 = a["poly_ring_off"][p0], a["poly_ring_off"][p1]
        v0, v1 = a["ring_off"][r0], a["ring_off"][r1]
        f0, f1 = a["fp_off"][i], a["fp_off"][i + 1]
        vs.append(a["verts"][v0:v1])
        fs.append(a["fp"][f0:f1])
        spans.append((p0, p1, r0, r1, v0, v1, f0, f1))
    V = np.concatenate(vs).astype(np.float64)
    Fp = np.concatenate(fs).astype(np.float64)
    pts = np.r_[V[:, :2], Fp] + sheet.origin
    lat, lon = from_puwg92(pts[:, 0], pts[:, 1])
    x, y, _ = frame.geodetic_to_enu(lat, lon, np.zeros(len(lat)))
    xy = np.c_[np.asarray(x, np.float64), np.asarray(y, np.float64)]
    xv, xf = xy[:len(V)], xy[len(V):]
    ov, of = 0, 0
    for i, (p0, p1, r0, r1, v0, v1, f0, f1) in zip(rows, spans):
        nv, nf = v1 - v0, f1 - f0
        verts = np.c_[xv[ov:ov + nv], V[ov:ov + nv, 2]]
        ring = xf[of:of + nf]
        ov, of = ov + nv, of + nf
        polys = []
        for p in range(p0, p1):
            ra, rb = a["poly_ring_off"][p], a["poly_ring_off"][p + 1]
            polys.append((int(a["poly_kind"][p]), [(int(a["ring_off"][r] - v0), int(a["ring_off"][r + 1] - v0))
                                                   for r in range(ra, rb)]))
        top = float(verts[:, 2].max())
        eave = _eave(verts, polys, top)
        if _area_xy(ring) < 0:
            ring = ring[::-1]
        out.append(Model(id=str(a["ids"][i]), ring=ring, verts=verts, polys=polys, roof_type=int(a["roof"][i]),
                         top=top, eave=max(eave, 0.0), ntri=int(a["ntri"][i]), flat=_flat_share(verts, polys)))
    return out


def models(frame, x0, y0, x1, y1, cache_dir, fetcher, kind="demand", margin=0.0):
    """The models (Model, in the anchor's frame) whose box meets the box [x0, x1] x [y0, y1] of the anchor's frame grown
    by margin; [] where there are none. Raises Unavailable when a sheet there could not be had now."""
    xs = np.array([x0 - margin, x1 + margin, x1 + margin, x0 - margin])
    ys = np.array([y0 - margin, y0 - margin, y1 + margin, y1 + margin])
    la, lo, _ = frame.enu_to_geodetic(xs, ys, np.zeros(4))
    la, lo = np.asarray(la, np.float64), np.asarray(lo, np.float64)
    if not covered(la.min(), lo.min(), la.max(), lo.max()):
        return []
    E, N = to_puwg92(la, lo)
    src = source(cache_dir, fetcher)
    out = []
    for name in sheets_in(la.min(), lo.min(), la.max(), lo.max()):
        s = src.sheet(name, kind)
        if s is None or not s.n:
            continue
        rows = s.select(E.min(), N.min(), E.max(), N.max())
        rows = [i for i in rows if s.a["ntri"][i] <= MAX_TRIS]
        out.extend(_models(s, rows, frame))
    return out
