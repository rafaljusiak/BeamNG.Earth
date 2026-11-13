"""pl_lod2 / pl_lod2_mesh: GUGiK's LoD2 models of Poland (EPSG:2180 and map sheets, the CityGML parse, the Range fetch
of one zip entry and its checks, the meshes, and their place among the map's footprints in tile_buildings). Offline:
a hand-written CityGML fixture and a fake fetcher."""
import io
import math
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock
import zipfile
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import buildings as bld  # noqa: E402
import geodesy  # noqa: E402
import netfetch  # noqa: E402
import pl_lod2 as P  # noqa: E402
import pl_lod2_mesh as PM  # noqa: E402

LAT, LON = 52.2400, 21.0900         # Grochow, Warsaw
E0, N0 = (float(v) for v in P.to_puwg92(LAT, LON))


def _poly(pts):
    pos = " ".join(f"{E0 + x:.3f} {N0 + y:.3f} {z:.3f}" for x, y, z in list(pts) + [pts[0]])
    return f"<gml:LinearRing><gml:posList srsDimension=\"3\">{pos}</gml:posList></gml:LinearRing>"


def _surface(kind, rings):
    ext = f"<gml:exterior>{_poly(rings[0])}</gml:exterior>"
    ints = "".join(f"<gml:interior>{_poly(r)}</gml:interior>" for r in rings[1:])
    return (f"<bldg:boundedBy><bldg:{kind}><bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember><gml:Polygon>"
            f"{ext}{ints}</gml:Polygon></gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface></bldg:{kind}>"
            f"</bldg:boundedBy>")


def _building(bid, roof_type, surfaces):
    body = "".join(_surface(k, r) for k, r in surfaces)
    return (f"<core:cityObjectMember><bldg:Building gml:id=\"g_{bid}\"><gen:stringAttribute name=\"buildingId\">"
            f"<gen:value>{bid}</gen:value></gen:stringAttribute><bldg:roofType>{roof_type}</bldg:roofType>{body}"
            f"</bldg:Building></core:cityObjectMember>")


def house(x=0.0, y=0.0, z=100.0):
    """A 10 x 8 m house, eaves 6 m, ridge 10 m along x (walls counter-clockwise seen from outside, as CityGML's)."""
    def p(*q):
        return [(x + a, y + b, z + c) for a, b, c in q]
    return [("GroundSurface", [p((0, 0, 0), (0, 8, 0), (10, 8, 0), (10, 0, 0))]),
            ("WallSurface", [p((0, 0, 0), (10, 0, 0), (10, 0, 6), (0, 0, 6))]),
            ("WallSurface", [p((10, 8, 0), (0, 8, 0), (0, 8, 6), (10, 8, 6))]),
            ("WallSurface", [p((10, 0, 0), (10, 8, 0), (10, 8, 6), (10, 4, 10), (10, 0, 6))]),
            ("WallSurface", [p((0, 8, 0), (0, 0, 0), (0, 0, 6), (0, 4, 10), (0, 8, 6))]),
            ("RoofSurface", [p((0, 0, 6), (10, 0, 6), (10, 4, 10), (0, 4, 10))]),
            ("RoofSurface", [p((0, 4, 10), (10, 4, 10), (10, 8, 6), (0, 8, 6))])]


def block(x=40.0, y=0.0, z=101.0):
    """A 20 x 20 m flat-roofed block, 15 m, round a 6 x 6 m courtyard (its roof's hole)."""
    def p(*q):
        return [(x + a, y + b, z + c) for a, b, c in q]
    return [("GroundSurface", [p((0, 0, 0), (0, 20, 0), (20, 20, 0), (20, 0, 0))]),
            ("WallSurface", [p((0, 0, 0), (20, 0, 0), (20, 0, 15), (0, 0, 15))]),
            ("WallSurface", [p((20, 0, 0), (20, 20, 0), (20, 20, 15), (20, 0, 15))]),
            ("WallSurface", [p((20, 20, 0), (0, 20, 0), (0, 20, 15), (20, 20, 15))]),
            ("WallSurface", [p((0, 20, 0), (0, 0, 0), (0, 0, 15), (0, 20, 15))]),
            ("RoofSurface", [p((0, 0, 15), (20, 0, 15), (20, 20, 15), (0, 20, 15)),
                             p((7, 7, 15), (7, 13, 15), (13, 13, 15), (13, 7, 15))])]


def citygml(blds):
    return ("<?xml version=\"1.0\" encoding=\"UTF-8\"?><core:CityModel xmlns:core=\"http://www.opengis.net/citygml/2.0\" "
            "xmlns:bldg=\"http://www.opengis.net/citygml/building/2.0\" xmlns:gml=\"http://www.opengis.net/gml\" "
            "xmlns:gen=\"http://www.opengis.net/citygml/generics/2.0\">" + "".join(blds) + "</core:CityModel>").encode()


FIXTURE = citygml([_building("H1", 1030, house()), _building("B1", 1130, block())])


class FakeFetcher:
    """Range requests of one in-memory zip; counts its requests."""

    def __init__(self, data, fail=False):
        self.data, self.fail, self.calls = data, fail, 0

    def get(self, url, kind="demand", headers=None, lane=None):
        self.calls += 1
        if self.fail:
            raise netfetch.Offline("offline", url, "x")
        a, b = (int(v) for v in headers["Range"].split("=")[1].split("-"))
        return netfetch.Response(206, self.data[a:b + 1], {})


def zipped(name, payload):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Modele_3D/README.txt", b"x" * 1000)
        z.writestr(name, payload)
    data = buf.getvalue()
    i = next(i for i in zipfile.ZipFile(io.BytesIO(data)).infolist() if i.filename == name)
    return data, [name.split("/")[-1][:4], i.header_offset, i.compress_size, i.file_size, i.CRC, i.compress_type]


class GeodesyTest(unittest.TestCase):
    def test_round_trip(self):
        lat, lon = np.meshgrid(np.linspace(49.0, 54.9, 13), np.linspace(14.1, 24.2, 13))
        e, n = P.to_puwg92(lat, lon)
        la, lo = P.from_puwg92(e, n)
        self.assertLess(np.abs(la - lat).max(), 1e-8)
        self.assertLess(np.abs(lo - lon).max(), 1e-8)

    def test_central_meridian(self):
        e, n = P.to_puwg92(52.0, 19.0)
        self.assertAlmostEqual(float(e), 500000.0, places=6)
        e1, n1 = P.to_puwg92(52.0, 19.01)
        # 0.01 degree of longitude at 52 N on the central meridian, times k0
        a, f = 6378137.0, 1 / 298.257222101
        nu = a / math.sqrt(1 - (2 * f - f * f) * math.sin(math.radians(52)) ** 2)
        self.assertAlmostEqual(float(e1 - e), 0.9993 * nu * math.cos(math.radians(52)) * math.radians(0.01), delta=0.01)


class SheetTest(unittest.TestCase):
    def test_palace_of_culture(self):
        self.assertEqual(P.sheet_of(52.2317, 21.0060), "N-34-139-A-c-1-1")
        s, w, n, e = P.sheet_bounds("N-34-139-A-c-1-1")
        self.assertTrue(s <= 52.2317 < n and w <= 21.0060 < e)
        self.assertAlmostEqual(n - s, 1.25 / 60)
        self.assertAlmostEqual(e - w, 1.875 / 60)

    def test_box_across_four_sheets(self):
        s, w, n, e = P.sheet_bounds("N-34-139-A-c-1-1")
        got = P.sheets_in(s + 0.001, w + 0.001, n + 0.001, e + 0.001)
        self.assertEqual(len(set(got)), 4)
        self.assertIn("N-34-139-A-c-1-1", got)
        self.assertIn("N-34-139-A-c-1-2", got)     # east
        self.assertIn("N-34-139-A-a-3-3", got)     # north: the 1:25,000 quarter above
        self.assertEqual(P.sheets_in(s + 0.001, w + 0.001, s + 0.002, w + 0.002), ["N-34-139-A-c-1-1"])

    def test_index_names(self):
        names = list(P.index()["sheets"])
        self.assertGreater(len(names), 30000)
        for name in names[::97]:
            s, w, n, e = P.sheet_bounds(name)
            self.assertEqual(P.sheet_of((s + n) / 2, (w + e) / 2), name)
        self.assertTrue(P.has_sheet("N-34-139-A-c-1-1"))
        self.assertTrue(P.covered(52.23, 21.00, 52.24, 21.01))
        self.assertFalse(P.covered(52.52, 13.40, 52.53, 13.41), "Berlin")


class ParseTest(unittest.TestCase):
    def setUp(self):
        self.blds = P.parse([FIXTURE[i:i + 333] for i in range(0, len(FIXTURE), 333)])
        self.frame = geodesy.EnuFrame(LAT, LON)

    def models(self):
        s = P.Sheet("X", P.pack(self.blds))
        return {m.id: m for m in P._models(s, s.select(E0 - 100, N0 - 100, E0 + 100, N0 + 100), self.frame)}

    def test_buildings(self):
        self.assertEqual([b[0] for b in self.blds], ["H1", "B1"])
        self.assertEqual([b[1] for b in self.blds], [1030, 1130])
        self.assertEqual(len(self.blds[0][4]), 6, "4 walls, 2 roofs (the ground outline apart)")
        self.assertEqual(len(self.blds[1][4][-1][1]), 2, "the block's roof: its outer ring and the courtyard")
        self.assertAlmostEqual(self.blds[1][2], 101.0)

    def test_models(self):
        m = self.models()
        h, b = m["H1"], m["B1"]
        self.assertAlmostEqual(h.eave, 6.0, places=3)
        self.assertAlmostEqual(h.top, 10.0, places=3)
        self.assertLess(h.flat, 0.01)
        self.assertGreater(b.flat, 0.99, "1130 (combined), but flat: the roof's slope decides")
        self.assertAlmostEqual(abs(bld._area2(h.ring)) / 2, 80.0, delta=0.5)
        self.assertGreater(bld._area2(h.ring), 0, "counter-clockwise")
        c = h.ring.mean(0)
        self.assertLess(math.hypot(c[0] - 5, c[1] - 4), 0.5, "at its place in the anchor's frame (grid convergence)")

    def test_duplicate_across_powiats(self):
        self.assertEqual(len(P.pack(self.blds + self.blds)["ids"]), 2)


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_entry(self):
        data, entry = zipped("Modele_3D/1465_N-34-139-A-c-1-1.gml", FIXTURE)
        f = FakeFetcher(data)
        blds = P.fetch_entry(f, "https://x/", entry, "N-34-139-A-c-1-1")
        self.assertEqual(len(blds), 2)
        self.assertEqual(f.calls, 1, "one Range request")

    def test_changed_zip(self):
        data, entry = zipped("Modele_3D/1465_N-34-139-A-c-1-1.gml", FIXTURE)
        with self.assertRaises(P.Unavailable):
            P.fetch_entry(FakeFetcher(data), "https://x/", entry[:4] + [entry[4] ^ 1] + entry[5:], "N-34-139-A-c-1-1")
        with self.assertRaises(P.Unavailable):
            P.fetch_entry(FakeFetcher(data), "https://x/", entry, "N-34-139-A-c-1-2")      # another sheet's name
        moved = list(entry)
        moved[1] += 7
        with self.assertRaises(P.Unavailable):
            P.fetch_entry(FakeFetcher(data), "https://x/", moved, "N-34-139-A-c-1-1")

    def test_source_caches(self):
        data, entry = zipped("Modele_3D/1465_N-34-139-A-c-1-1.gml", FIXTURE)
        f = FakeFetcher(data)
        idx = {"base": "https://x/", "zips": {}, "sheets": {"N-34-139-A-c-1-1": [entry]}}
        with unittest.mock.patch.object(P, "index", lambda: idx):
            s = P.Source(self.tmp, f).sheet("N-34-139-A-c-1-1")
            self.assertEqual(s.n, 2)
            s2 = P.Source(self.tmp, FakeFetcher(b"", fail=True)).sheet("N-34-139-A-c-1-1")
            self.assertEqual(s2.n, 2, "from the disk")
            self.assertIsNone(P.Source(self.tmp, f).sheet("N-34-139-A-c-1-2"), "no models there")
            off = FakeFetcher(b"", fail=True)
            src = P.Source(os.path.join(self.tmp, "b"), off)
            with self.assertRaises(P.Unavailable):
                src.sheet("N-34-139-A-c-1-1")
            with self.assertRaises(P.Unavailable):
                src.sheet("N-34-139-A-c-1-1")
            self.assertEqual(off.calls, 1, "not asked again at once")

    def test_offline_tile_keeps_footprints(self):
        import revisit

        class MD:
            cache_dir = self.tmp
            fetcher = FakeFetcher(b"", fail=True)
        frame = geodesy.EnuFrame(52.2317, 21.0060)
        n0 = revisit.faults()
        with unittest.mock.patch.object(P, "ON", True):
            self.assertIsNone(bld.lod2_models(frame, 0, 0, 256, 256, MD()))
        self.assertGreater(revisit.faults(), n0, "the build is incomplete: made again next visit")
        with unittest.mock.patch.object(P, "ON", False):
            self.assertIsNone(bld.lod2_models(frame, 0, 0, 256, 256, MD()), "off: not asked")


class MeshTest(unittest.TestCase):
    def setUp(self):
        blds = P.parse([FIXTURE])
        frame = geodesy.EnuFrame(LAT, LON)
        s = P.Sheet("X", P.pack(blds))
        self.m = {m.id: m for m in P._models(s, np.arange(s.n), frame)}

    def spec(self, m, cls="apartments", nf=None):
        at = bld._Lod2([m], 0, 0).attrs(0)
        at.update({"class": cls, "num_floors": nf})
        spec = bld.plan(m.ring, at, lat=LAT, lon=LON)
        spec.lod2 = m
        return spec

    def mesh(self, spec, collision=False):
        if collision:
            r = bld.collision_mesh([(spec, 0.0, 1.0)])
        else:
            r = bld.build([(spec, 0.0, 1.0)])
        return r[0], r[2]

    def test_walls_out_roofs_up(self):
        for mid in ("H1", "B1"):
            spec = self.spec(self.m[mid])
            P_, T = self.mesh(spec)
            a, b, c = P_[T[:, 0]], P_[T[:, 1]], P_[T[:, 2]]
            n = np.cross(b - a, c - a)
            cen = (a + b + c) / 3
            mid_xy = self.m[mid].ring.mean(0)
            wall = np.abs(n[:, 2]) < 1e-6 * np.linalg.norm(n, axis=1).max()
            outward = ((cen[wall, :2] - mid_xy) * n[wall, :2]).sum(1)
            self.assertTrue((outward > 0).all(), f"{mid}: every wall triangle faces out")
            self.assertTrue((n[~wall, 2] > 0).all(), f"{mid}: roofs face up")
            self.assertAlmostEqual(P_[:, 2].max(), 1.0 + self.m[mid].top, places=3, msg="on the ground's highest")
            self.assertAlmostEqual(P_[:, 2].min(), 0.0 - bld.SKIRT_M, places=3, msg="the walls reach under it")

    def test_roof_area_and_courtyard(self):
        for mid, want in (("H1", 80.0), ("B1", 400.0 - 36.0)):
            m = bld.Mesh()
            PM._roofs(m, self.m[mid], 0.0, (0, (1, 1, 1, 1)))
            P_, _, T, _, _ = m.arrays()
            u, w = P_[T[:, 1]] - P_[T[:, 0]], P_[T[:, 2]] - P_[T[:, 0]]
            self.assertAlmostEqual(np.abs(u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]).sum() / 2, want, delta=0.5, msg=mid)

    def test_floors_and_bays(self):
        spec = self.spec(self.m["B1"], nf=5)
        self.assertEqual(spec.floors, 5)
        self.assertEqual(spec.roof, "flat")
        P_, T = self.mesh(spec)
        zs = np.unique(np.round(P_[:, 2], 3))
        for f in range(1, 5):
            self.assertIn(round(1.0 + 3.0 * f, 3), zs, "a cut at every storey")
        house = self.spec(self.m["H1"], cls="house", nf=2)
        self.assertNotEqual(house.roof, "flat")
        self.assertEqual(house.floors, 2, "two storeys under the 6 m eaves")

    def test_collision(self):
        P_, T = self.mesh(self.spec(self.m["H1"]), collision=True)
        self.assertLess(len(T), 30)
        self.assertAlmostEqual(P_[:, 2].min(), 0.0 - bld.FOUNDATION_M, places=3)
        self.assertAlmostEqual(P_[:, 2].max(), 11.0, places=3)

    def test_bridged_hole(self):
        outer = np.array([(0, 0), (20, 0), (20, 20), (0, 20)], float)
        holes = [np.array([(5, 5), (8, 5), (8, 8), (5, 8)], float), np.array([(12, 12), (15, 12), (15, 15), (12, 15)], float)]
        ring = PM.bridge(outer, holes)
        t = PM.triangulate(ring)
        u, w = ring[t[:, 1]] - ring[t[:, 0]], ring[t[:, 2]] - ring[t[:, 0]]
        self.assertAlmostEqual((u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]).sum() / 2, 400 - 18, places=6)


class TileTest(unittest.TestCase):
    """tile_buildings with models: a footprint mostly inside one is replaced and lends it its attributes, one outside
    stays, a landmark's or a canopy's takes it away."""

    def setUp(self):
        self.frame = geodesy.EnuFrame(LAT, LON)
        s = P.Sheet("X", P.pack(P.parse([FIXTURE])))
        self.models = P._models(s, np.arange(s.n), self.frame)

    def ll(self, x, y, w, d):
        e = np.array([x, x + w, x + w, x, x]); n = np.array([y, y, y + d, y + d, y])
        la, lo, _ = self.frame.enu_to_geodetic(e, n, 0.0)
        return np.stack([lo, la], 1)

    def items(self, feats, skip=frozenset(), models=True):
        from test_buildings import Feats
        ids, rings, cls, nf, col = zip(*feats)
        b = Feats(list(ids), list(rings), **{"class": list(cls), "num_floors": list(nf), "height": [np.nan] * len(ids),
                                             "facade_color": list(col)})
        return bld.tile_buildings(self.frame, -128, -128, 128, 128, [b], lambda la, lo: np.zeros(np.shape(la)),
                                  skip=skip, lod2_models=self.models if models else None)

    def test_replaced_and_lent(self):
        items = self.items([("a", self.ll(0.3, 0.2, 9.5, 7.6), "house", 2.0, "#e0d0b0"),
                            ("b", self.ll(40, 0, 10, 20), "apartments", 4.0, ""),       # half the block ...
                            ("c", self.ll(50, 0, 10, 20), "retail", 1.0, ""),           # ... and the other half
                            ("d", self.ll(-60, -60, 12, 10), "house", 1.0, "")])        # newer: no model
        mods = {s.lod2.id: s for s, _, _ in items if s.lod2 is not None}
        self.assertEqual(sorted(mods), ["B1", "H1"])
        self.assertEqual(len(items), 3, "two models and the house no model covers")
        self.assertEqual(mods["H1"].cls, "house")
        self.assertEqual(mods["H1"].floors, 2)
        self.assertIn("|c_", mods["H1"].facade, "the footprint's colour")
        self.assertIn(mods["B1"].cls, ("apartments", "retail"), "the larger share's class (a tie: the name decides)")
        self.assertEqual(mods["B1"].roof, "flat")
        self.assertEqual(mods["H1"].ring.shape, self.models[0].ring.shape)

    def test_look_keyed_on_the_donor(self):
        # a model's look is keyed on its largest donor (the segments of one mapped block, a model each, look alike in
        # every tile), else on its own id
        items = self.items([("whole", self.ll(39, -1, 22, 22), "apartments", np.nan, "")])
        seeds = {s.lod2.id: s.seed for s, _, _ in items if s.lod2 is not None}
        self.assertEqual(seeds["B1"], bld._hash01("lod2:whole", "plan"))
        self.assertEqual(seeds["H1"], bld._hash01("lod2:H1", "plan"))

    def test_off_and_canopy(self):
        feats = [("a", self.ll(0.3, 0.2, 9.5, 7.6), "house", 2.0, "")]
        self.assertTrue(all(s.lod2 is None for s, _, _ in self.items(feats, models=False)))
        items = self.items(feats, skip=frozenset({"a"}))
        self.assertNotIn("H1", [s.lod2.id for s, _, _ in items if s.lod2 is not None], "no model under a canopy")

    def test_partly_inside_stays(self):
        items = self.items([("a", self.ll(5, 0, 20, 8), "house", 1.0, "")])      # a quarter on the house
        self.assertIn("house", [s.cls for s, _, _ in items if s.lod2 is None])


if __name__ == "__main__":
    unittest.main()
