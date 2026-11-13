"""Sidewalks mapped as ways of their own beside a road (walksetback.py, roads._osm_sidewalks, roads.walk_at): Lublin's
Puchacza at 51.2335 N 22.5880 E, sidewalk=separate on nearly every way and a footway two metres behind a strip of
grass, was a bare street ("the sidewalk is rendered as asphalt or road", the user, 2026-11-13)."""
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import junctions  # noqa: E402
import roads  # noqa: E402
import walksetback  # noqa: E402
from test_roads import features, flat, FRAME  # noqa: E402

TOWN = lambda lat, lon: np.ones(np.shape(lat), bool)         # noqa: E731
STREET = [(0, 0), (100, 0)]
TOL = 0.05      # m (the fixture's metres a degree are round numbers)


def street(paths, tag="separate", classes=None, cls="residential", **kw):
    """A street along +x (its left: +y) and the paths beside it; the street's sidewalk tag."""
    f = features([STREET] + [p for p in paths], [cls] + list(classes or ["footway"] * len(paths)), **kw)
    f.attrs["osm_sidewalk"] = np.array([tag] + [""] * len(paths), object)
    return f


def along(y, x0=0.0, x1=100.0, step=10.0):
    return [(float(x), float(y)) for x in np.arange(x0, x1 + 0.01, step)]


class MeasureTest(unittest.TestCase):
    def measure(self, f, half=3.0, **kw):
        n = len(f)
        return walksetback.measure(f, [0], list(range(1, n)), {0: half}, **kw)

    def test_a_footway_along_the_street_is_found_on_its_side(self):
        got = self.measure(street([along(6.0), along(-7.5)]))
        self.assertEqual(sorted(got[0]), [-1, 1])
        self.assertAlmostEqual(got[0][1][0], 6.0, delta=TOL)
        self.assertAlmostEqual(got[0][-1][0], 7.5, delta=TOL)
        self.assertEqual(got[0][1][1], walksetback.WIDTH_M)

    def test_the_nearer_of_two_and_its_mapped_width(self):
        got = self.measure(street([along(9.0), along(5.0)]), path_w={2: 3.0})
        self.assertAlmostEqual(got[0][1][0], 5.0, delta=TOL)
        self.assertEqual(got[0][1][1], 3.0)

    def test_not_a_path_across_far_off_short_or_wandering(self):
        across = [(50.0, -30.0), (50.0, 30.0)]
        far = along(3.0 + walksetback.REACH_M + 1.0)
        short = along(6.0, 0.0, 40.0)                                   # beside 40 m of the street's 100
        wandering = [(0.0, 4.0), (100.0, 4.0 + walksetback.SPREAD_M * 2)]
        slanted = [(x, -4.0 - 0.6 * (x - 40.0)) for x in (40.0, 45.0, 50.0)]     # 31 degrees off the street
        for name, path in (("across", across), ("far", far), ("short", short), ("wandering", wandering),
                           ("slanted", slanted)):
            self.assertEqual(self.measure(street([path])), {}, name)
        self.assertEqual(self.measure(street([along(2.0)])), {}, "a line inside the carriageway")

    def test_the_junctions_at_the_ends_do_not_count(self):
        # the footway turns away with the side street 5 m before the street's end: still the street's
        got = self.measure(street([along(6.0, 5.0, 95.0) + [(96.0, 30.0)]]))
        self.assertAlmostEqual(got[0][1][0], 6.0, delta=TOL)

    def test_many_roads_at_once_as_one_by_one(self):
        lines = [[(0, 0), (100, 0)], [(0, 40), (60, 40), (60, 100)], [(200, 0), (200, 9)]]
        paths = [along(5.5), along(-6.0, 20.0, 90.0), [(0.0, 45.0), (55.0, 45.0), (55.0, 100.0)],
                 [(0.0, 34.0), (66.0, 34.0), (66.0, 100.0)], [(204.5, 0.0), (204.5, 9.0)]]
        f = features(lines + paths, ["residential"] * 3 + ["footway"] * 5)
        half = {0: 3.0, 1: 3.0, 2: 3.0}
        every = walksetback.measure(f, [0, 1, 2], list(range(3, 8)), half)
        for i in (0, 1, 2):
            self.assertEqual(walksetback.measure(f, [i], list(range(3, 8)), half), {i: every[i]} if i in every else {})
        self.assertEqual(sorted(every[0]), [-1, 1])
        self.assertEqual(sorted(every[1]), [-1, 1])     # (its left along +x, its right after the turn north)
        self.assertAlmostEqual(every[1][1][0], 5.0, delta=TOL)
        self.assertAlmostEqual(every[1][-1][0], 6.0, delta=TOL)
        self.assertAlmostEqual(every[2][-1][0], 4.5, delta=TOL)        # a 9 m piece between two junctions


class SpecTest(unittest.TestCase):
    def spec(self, f, urban=TOWN):
        return roads.road_specs(f, urban_fn=urban)[0]

    def test_separate_is_a_sidewalk_at_the_footway(self):
        sp = self.spec(street([along(7.0)]))
        self.assertTrue(roads.has_sidewalks(sp))
        self.assertEqual(roads.walk_sides(sp), (1,), "mapped along one side: that side's")
        (side, at, w), = sp.sep
        self.assertEqual((side, w), (1, walksetback.WIDTH_M))
        self.assertAlmostEqual(at, 7.0, delta=TOL)
        gap, w = roads.walk_at(sp, 1)
        self.assertAlmostEqual(gap, 7.0 - walksetback.WIDTH_M / 2 - sp.width / 2, delta=TOL)       # the verge
        self.assertEqual(w, walksetback.WIDTH_M)
        self.assertAlmostEqual(roads.walk_reach(sp, 1), 7.0 + walksetback.WIDTH_M / 2 - sp.width / 2, delta=TOL)
        both = self.spec(street([along(7.0), along(-6.0)]))
        self.assertEqual(roads.walk_sides(both), (1, -1))
        self.assertEqual([s for s, _, _ in both.sep], [-1, 1])

    def test_separate_without_a_footway_beside_it_is_the_curb_sidewalk(self):
        for name, paths in (("none", []), ("far", [along(20.0)])):
            for urban in (TOWN, None):          # (the map says there is one: in the land cover's countryside too)
                sp = self.spec(street(paths), urban)
                self.assertTrue(roads.has_sidewalks(sp), name)
                self.assertEqual(roads.walk_sides(sp), (1, -1), name)
                self.assertIsNone(sp.sep, name)
                self.assertEqual(roads.walk_at(sp, 1), (0.0, sp.walk), name)

    def test_a_footway_at_the_curb_is_the_curb_sidewalk(self):
        sp0 = self.spec(street([]))
        sp = self.spec(street([along(sp0.width / 2 + 1.1)]))    # 10 cm of verge: none
        self.assertEqual(roads.walk_sides(sp), (1,))
        self.assertEqual(roads.walk_at(sp, 1), (0.0, sp.walk))

    def test_an_untagged_street_keeps_its_rule_and_takes_the_footways_place(self):
        sp = self.spec(street([along(-7.0)], tag=""))
        self.assertIsNone(sp.sides)
        self.assertEqual(roads.walk_sides(sp), (1, -1))
        self.assertEqual(roads.walk_at(sp, 1), (0.0, sp.walk))                  # no footway: at the curb
        self.assertAlmostEqual(roads.walk_at(sp, -1)[0], 7.0 - 1.0 - sp.width / 2, delta=TOL)
        rural = self.spec(street([along(-7.0)], tag=""), None)
        self.assertFalse(roads.has_sidewalks(rural), "a path beside a country road makes no sidewalk")
        self.assertIsNone(rural.sep)

    def test_tagged_sides_and_no_are_the_tags(self):
        for tag, sides in (("both", (1, -1)), ("left", (1,)), ("no", ())):
            sp = self.spec(street([along(7.0), along(-7.0)], tag=tag))
            self.assertEqual((sp.sides, sp.sep), (sides, None), tag)

    def test_only_built_paths_and_not_crossings(self):
        self.assertIsNone(self.spec(street([along(7.0)], surfaces=[(1, "ground")], classes=["path"])).sep)
        self.assertIsNotNone(self.spec(street([along(7.0)], surfaces=[(1, "paving_stones")], classes=["path"])).sep)
        self.assertIsNotNone(self.spec(street([along(7.0)], classes=["cycleway"])).sep)
        self.assertIsNone(self.spec(street([along(7.0)], classes=["steps"])).sep)
        self.assertIsNone(self.spec(street([along(7.0)], classes=["service"])).sep)

    def test_the_footways_mapped_width(self):
        sp = self.spec(street([along(7.0)], widths=[(1, 3.5)]))
        self.assertEqual(sp.sep[0][2], 3.5)
        self.assertEqual(self.spec(street([along(7.0)], widths=[(1, 14.0)])).sep[0][2], walksetback.WIDTH_M)

    def test_a_wide_pavement_is_drawn_no_wider_than_a_sidewalk(self):
        self.assertEqual(self.spec(street([along(7.0)], widths=[(1, 5.5)])).sep[0][2], walksetback.WIDTH_MAX_M)

    def test_a_footway_behind_the_measured_building_fronts_is_not_the_streets(self):
        # (roadwidth.py: the carriageway and the sidewalk between the buildings; a footway past them: an arcade)
        f = street([along(7.0)])
        narrow = roads.road_specs(f, TOWN, lambda fid, ll: (6.0, 1.5))[0]
        self.assertEqual((narrow.walk, narrow.sep), (1.5, None))
        self.assertEqual(roads.walk_at(narrow, 1), (0.0, 1.5))
        f = street([along(7.0)])
        wide = roads.road_specs(f, TOWN, lambda fid, ll: (6.0, 6.0))[0]
        self.assertIsNotNone(wide.sep)

    def test_an_expressways_carriageway_gets_none_from_the_tag_alone(self):
        f = street([], cls="primary")
        fast = roads.Spec(7.0, "paved", True, "primary", True, kmh=90.0, dir=1)
        self.assertFalse(roads.has_sidewalks(roads._osm_sidewalks(f, [fast])[0]))
        self.assertTrue(roads.has_sidewalks(roads._osm_sidewalks(f, [fast._replace(kmh=50.0)])[0]))

    def test_off_is_the_curb_sidewalk(self):
        with mock.patch.object(roads, "WALK_SETBACK", False):
            sp = self.spec(street([along(7.0)]))
        self.assertEqual((sp.sides, sp.sep), ((1, -1), None))

    def test_specs_round_trip_as_json(self):
        import json
        sp = self.spec(street([along(7.0), along(-6.0)]))
        back = roads.spec_from(json.loads(json.dumps(roads._spec_json(sp))))
        self.assertEqual(back, sp)
        hash(back)
        old = {k: v for k, v in roads._spec_json(self.spec(street([]))).items() if k != "sep"}     # (facts 7)
        self.assertIsNone(roads.spec_from(old).sep)

    def test_a_bridge_keeps_none(self):
        f = features([[(-100, 0), (0, 0)], [(0, 0), (100, 0)]], ["residential"] * 2, flags=[(1, "is_bridge")])
        f.attrs["osm_sidewalk"] = np.array(["separate", "separate"], object)
        self.assertIsNone(roads.deck_walks(f, 1, roads.road_specs(f, urban_fn=TOWN)))
        self.assertIsNone(roads.deck_walks(f, 0, roads.road_specs(f, urban_fn=TOWN)))


class TileTest(unittest.TestCase):
    def tile(self, f, **kw):
        walks = []
        chunks = roads.build_tile(FRAME, 0, 0, [f], flat, TOWN, walks=walks, **kw)
        return chunks, [w for w in walks if "pts" in w]

    def test_the_run_is_on_the_footways_line_and_the_gutter_at_the_curb(self):
        f = street([along(7.0, 0.0, 200.0, 20.0)])
        f.lonlat[1, 0] = f.lonlat[0, 0] + (f.lonlat[1, 0] - f.lonlat[0, 0]) * 2          # the street: 200 m
        chunks, walks = self.tile(f)
        self.assertEqual(len(walks), 1)
        wk = walks[0]
        sp = roads.road_specs(f, urban_fn=TOWN)[0]
        np.testing.assert_allclose(wk["pts"][:, 1], 7.0, atol=0.05)
        self.assertEqual((wk["side"], wk["w"]), (1, walksetback.WIDTH_M))
        self.assertAlmostEqual(wk["gap"], 7.0 - 1.0 - sp.width / 2, delta=TOL)
        gutters = [c for c in chunks if c["m"] == roads.GUTTER["m"]]
        self.assertTrue(gutters and all(abs(abs(n[1]) - (sp.width / 2 - roads.GUTTER["w"] / 2)) < 0.05
                                        for c in gutters for n in c["n"]))
        # the storm drains in the gutter, not in the verge by the sidewalk
        drains = [c for c in chunks if c["m"] == roads.DRAIN["m"]]
        self.assertTrue(drains and all(abs(n[1] - (sp.width / 2 - roads.GUTTER["w"] / 2)) < 0.05
                                       for c in drains for n in c["n"]))

    def test_a_side_streets_sidewalk_stops_at_the_far_sidewalks_outer_edge(self):
        # a wider street with its sidewalk 7 m out on the left and at the curb on the right; a side street across it
        f = features([[(-200, 100), (200, 100)], [(0, -100), (0, 300)], along(107.0, -200.0, 200.0, 20.0)],
                     ["secondary", "residential", "footway"])
        f.attrs["osm_sidewalk"] = np.array(["separate", "both", ""], object)
        main = roads.road_specs(f, urban_fn=TOWN)[0]
        self.assertEqual(roads.walk_sides(main), (1,))
        _, walks = self.tile(f)
        side = [w for w in walks if w["fid"] == "id001"]
        north = min(w["pts"][:, 1].min() for w in side if w["pts"][:, 1].mean() > 100)
        south = max(w["pts"][:, 1].max() for w in side if w["pts"][:, 1].mean() < 100)
        self.assertAlmostEqual(north, 100 + 7.0 + walksetback.WIDTH_M / 2, delta=0.7)   # past the verge and the walk
        self.assertAlmostEqual(south, 100 - main.width / 2, delta=0.7)                  # no sidewalk there: the curb

    def test_the_corners_slab_reaches_the_sidewalks_outer_edge(self):
        sp = roads.road_specs(street([along(7.0)]), urban_fn=TOWN)[0]
        self.assertAlmostEqual(junctions._walk_reach(sp, 1), 8.0 - sp.width / 2, delta=TOL)
        self.assertEqual(junctions._walk_reach(sp._replace(sep=None), 1), sp.walk)
        self.assertEqual(junctions._walk_reach(sp, -1), sp.walk)

    def test_parked_cars_stand_at_the_curb(self):
        import parking
        f = street([along(9.0, 0.0, 200.0, 20.0), along(-9.0, 0.0, 200.0, 20.0)], cls="secondary")
        f.lonlat[1, 0] = f.lonlat[0, 0] + (f.lonlat[1, 0] - f.lonlat[0, 0]) * 2
        park = []
        with mock.patch.object(parking, "STREET_SHARE", 2.0):          # (every street has parked cars)
            roads.build_tile(FRAME, 0, 0, [f], flat, TOWN, walks=[], props=[], park=park)
        sp = roads.road_specs(f, urban_fn=TOWN)[0]
        self.assertTrue(len(park))
        for row in park:
            self.assertLess(abs(row[1]), sp.width / 2 + 0.1, "in the carriageway by the curb, not on the verge")


class GroundTest(unittest.TestCase):
    def test_the_verge_is_not_painted_as_the_street(self):
        # the ground's material was the street's out to the sidewalk's outer edge: asphalt under the grass strip
        import surface as sf
        from test_roads import LAT, LON, M_LAT, M_LON
        n = 128
        lat = LAT + (n / 2 - np.arange(n)[:, None]) / M_LAT + np.zeros((1, n))
        lon = LON + (np.arange(n)[None, :] - n / 2) / M_LON + np.zeros((n, 1))
        f = street([along(10.0, -200.0, 200.0, 50.0)])
        f.lonlat[0, 0] = LON - 200 / M_LON
        f.lonlat[1, 0] = LON + 200 / M_LON
        sp = roads.road_specs(f)[0]
        self.assertEqual(roads.walk_sides(sp), (1,))
        gap, w = roads.walk_at(sp, 1)
        self.assertGreater(gap, 3.0)
        col = [int(v) for v in sf.paint_roads(np.full(lat.shape, sf.GRASS, np.uint8), lat, lon, [f], 1.0)[:, 10]]
        north = lambda m: col[64 - int(m)]                                  # noqa: E731 (the square m..m + 1 north)
        self.assertEqual(north(0), sf.ASPHALT)
        self.assertEqual(north(sp.width / 2 + gap / 2), sf.GRASS, "the verge")
        self.assertEqual(north(sp.width / 2 + gap + w / 2), sf.ASPHALT, "under the sidewalk: no grass through it")
        self.assertEqual(north(sp.width / 2 + gap + w + 3), sf.GRASS)
        self.assertEqual(col[64 + int(sp.width / 2 + 3)], sf.GRASS, "no sidewalk on the other side")


if __name__ == "__main__":
    unittest.main()
