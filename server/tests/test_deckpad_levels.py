"""Junction pads connect road surfaces at the same height, not merely the same XY endpoint."""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fast  # noqa: E402
import roads  # noqa: E402
import roadmesh  # noqa: E402


@unittest.skipUnless(fast.AVAILABLE and getattr(fast._ef, "VERSION", 0) >= 113,
                     "earthfast 113 (deck_pad_rim)")
class DeckPadLevelsTest(unittest.TestCase):
    @staticmethod
    def deck(fid, end, z, end_z=None):
        return (fid, SimpleNamespace(width=8.0), np.array([[0.0, 0.0], end]),
                np.array([z, z if end_z is None else end_z], dtype=np.float64))

    @staticmethod
    def pads(decks, ground=(), ground_z=0.0):
        return roads.deck_pads(decks, ground, (-50.0, -50.0, 50.0, 50.0),
                               lambda xy: np.full(len(xy), ground_z))

    def test_stacked_endpoints_do_not_form_a_ramp_between_levels(self):
        low = self.deck("low", [30.0, 0.0], 10.0)
        high = self.deck("high", [0.0, 30.0], 20.0)
        self.assertEqual(self.pads([low, high]), [])

    def test_two_junctions_above_each_other_keep_separate_surfaces(self):
        decks = [self.deck("a", [30.0, 0.0], 10.0), self.deck("b", [0.0, 30.0], 10.0),
                 self.deck("c", [-30.0, 0.0], 20.0), self.deck("d", [0.0, -30.0], 20.0)]
        for order in (decks, decks[::-1], [decks[i] for i in (0, 2, 1, 3)]):
            pads = self.pads(order)
            self.assertEqual(len(pads), 2)
            self.assertEqual(sorted(float(p[1][2]) for p in pads), [10.0, 20.0])
            for ring, centre, _ in pads:
                np.testing.assert_allclose(ring[:, 2], centre[2], atol=1e-9)

    def test_ground_endpoint_below_a_bridge_does_not_join_it(self):
        bridge = self.deck("bridge", [30.0, 0.0], 20.0)
        street = ("street", SimpleNamespace(width=8.0), np.array([[0.0, 0.0], [0.0, 30.0]]), None)
        self.assertEqual(self.pads([bridge], [street], ground_z=0.0), [])
        self.assertEqual(len(self.pads([bridge], [street], ground_z=20.0)), 1,
                         "a physical approach at the deck height still joins")

    def test_connected_ramps_keep_their_grades_and_junction(self):
        # Their remote ends differ by metres; only the shared endpoint determines connection.
        decks = [self.deck("up", [30.0, 0.0], 20.0, 22.0),
                 self.deck("down", [0.0, 30.0], 20.0, 18.0),
                 self.deck("approach", [-30.0, -20.0], 20.0, 19.0)]
        before = [d[3].copy() for d in decks]
        pads = self.pads(decks)
        self.assertEqual(len(pads), 1)
        self.assertEqual(float(pads[0][1][2]), 20.0)
        self.assertGreater(float(np.ptp(pads[0][0][:, 2])), 0.5, "the junction is not flattened")
        for d, z in zip(decks, before):
            np.testing.assert_array_equal(d[3], z)

    def test_height_matching_does_not_chain_across_several_levels(self):
        decks = [self.deck("a", [30.0, 0.0], 10.0), self.deck("b", [0.0, 30.0], 10.6),
                 self.deck("c", [-30.0, 0.0], 11.2)]
        pads = self.pads(decks)
        self.assertEqual(len(pads), 1)
        self.assertAlmostEqual(float(pads[0][1][2]), 10.3)
        self.assertLessEqual(float(pads[0][0][:, 2].max()), 10.6)

    def test_one_ground_approach_cannot_bridge_two_height_groups(self):
        decks = [self.deck("low", [30.0, 0.0], 10.0),
                 self.deck("high", [-30.0, 0.0], 11.2)]
        street = ("street", SimpleNamespace(width=8.0), np.array([[0.0, 0.0], [0.0, 30.0]]), None)
        pads = self.pads(decks, [street], ground_z=10.6)
        self.assertEqual(len(pads), 1, "the one approach belongs to at most one physical junction")
        self.assertLessEqual(float(np.ptp(pads[0][0][:, 2])), roads.PAD_JOIN_Z_M)

    def test_missing_ground_elevation_does_not_make_an_invalid_pad(self):
        bridge = self.deck("bridge", [30.0, 0.0], 20.0)
        street = ("street", SimpleNamespace(width=8.0), np.array([[0.0, 0.0], [0.0, 30.0]]), None)
        self.assertEqual(self.pads([bridge], [street], ground_z=np.nan), [])

    def test_a_lower_pad_does_not_remove_upper_bridge_guardrails(self):
        deck = {'fid': 'upper', 'deck': True, 'fixed': True, 'w': 8., 'm': 'road_asphalt_2lane',
                'n': [[-25., 0., 20.], [25., 0., 20.]], 'tl': 10.}
        ring = np.array([[-10., -8., 10.], [10., -8., 10.], [10., 8., 10.], [-10., 8., 10.]])
        baseline = roadmesh.parapet_runs([deck])
        result = roadmesh.parapet_runs([deck], pads=[(ring, np.array([0., 0., 10.]))])
        def length(runs):
            return sum(np.linalg.norm(np.diff(r['edge'], axis=0), axis=1).sum() for r in runs)
        self.assertAlmostEqual(length(result), length(baseline), places=9)
        self.assertEqual(len(result), len(baseline))

    def test_pad_height_is_interpolated_at_the_edge_point(self):
        # A long steep pad spans the upper bridge's height somewhere else, but not at this point.
        ring = np.array([[-10., -8., 10.], [10., -8., 20.], [10., 8., 20.], [-10., 8., 10.]])
        pad = (ring, np.array([0., 0., 15.]))
        xy = np.array([[-8., 0.], [8., 0.], [0., 0.]])
        np.testing.assert_array_equal(roads.pad_inside([pad], xy, z=np.array([20., 19., 15.])),
                                      [False, True, True])
        self.assertTrue(roads.pad_inside([pad], xy).all(), 'legacy XY-only query stays available')


@unittest.skipUnless(fast.AVAILABLE and getattr(fast._ef, "VERSION", 0) >= 113,
                     "earthfast 113 (deck_pad_rim)")
class DeckPadRimOnPavingTest(unittest.TestCase):
    """A pad's rim has no railing where the paving goes on: on another deck's carriageway, inside another pad
    (Warsaw, Aleja Niepodleglosci over the Trasa Lazienkowska's tunnel: a ring of short links on decks)."""

    @staticmethod
    def deck(fid, a, b, z=20.0, width=8.0):
        return (fid, SimpleNamespace(width=width), np.array([a, b], dtype=np.float64), np.array([z, z]))

    @staticmethod
    def pads(decks):
        return roads.deck_pads(decks, (), (-200.0, -200.0, 200.0, 200.0), lambda xy: np.zeros(len(xy)))

    @staticmethod
    def wall_points(pad):
        return np.concatenate([w[0] for w in pad[2]]) if pad[2] else np.zeros((0, 3))

    def tee(self):
        return [self.deck("w", [0.0, 0.0], [-40.0, 0.0]), self.deck("e", [0.0, 0.0], [10.0, 0.0]),
                self.deck("n", [0.0, 0.0], [0.0, 40.0])]

    def test_a_lone_junction_keeps_its_railings(self):
        pads = self.pads(self.tee())
        self.assertEqual(len(pads), 1)
        self.assertGreater(len(self.wall_points(pads[0])), 4)

    def test_no_railing_on_the_deck_beside_or_in_the_next_pad(self):
        # two T junctions 10 m apart on one street: each pad's rim reaches the other's stem and pad
        decks = self.tee() + [self.deck("e2", [10.0, 0.0], [50.0, 0.0]), self.deck("s", [10.0, 0.0], [10.0, -40.0])]
        pads = self.pads(decks)
        self.assertEqual(len(pads), 2)
        surfaces = [(d[2], d[3], d[1].width / 2) for d in decks]
        for n, pad in enumerate(pads):
            pts = self.wall_points(pad)
            self.assertGreater(len(pts), 2, "the outer rim over the drop keeps its railing")
            self.assertFalse(roads._on_decks(pts, surfaces).any(), "a railing across a deck's lanes")
            self.assertFalse(roads.pad_inside([pads[1 - n]], pts[:, :2], z=pts[:, 2]).any(),
                             "a railing inside the next pad")
        alone = self.pads(self.tee())[0]
        self.assertLess(len(self.wall_points(pads[0])), len(self.wall_points(alone)))

    def test_a_deck_passing_under_takes_no_railing_away(self):
        tee = self.tee()
        under = self.deck("under", [-6.0, -40.0], [-6.0, 40.0], z=12.0)
        want = self.wall_points(self.pads(tee)[0])
        got = [p for p in self.pads(tee + [under]) if abs(p[1][2] - 20.0) < 0.1]
        self.assertEqual(len(got), 1)
        np.testing.assert_allclose(self.wall_points(got[0]), want)

    def test_on_decks_is_by_width_and_height(self):
        s = [(np.array([[0.0, 0.0], [20.0, 0.0]]), np.array([10.0, 12.0]), 4.0)]
        pts = np.array([[10.0, 3.9, 11.0], [10.0, 4.1, 11.0], [10.0, 0.0, 12.0], [10.0, 0.0, 5.0], [25.0, 0.0, 12.0]])
        self.assertEqual(roads._on_decks(pts, s).tolist(), [True, False, False, False, False])


if __name__ == "__main__":
    unittest.main()
