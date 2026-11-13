"""Sidewalks mapped on their own beside a road (the user, 2026-11-13, Lublin's Puchacza at 51.2335 N 22.5880 E: "the
sidewalk is rendered as asphalt or road").

OpenStreetMap maps a sidewalk either as a tag of its road (sidewalk=both / left / right) or as a way of its own beside
it (highway=footway, with or without footway=sidewalk; the road then says sidewalk=separate, or sidewalk:left / :right
= separate, which the Layercake file does not carry, or nothing at all). Poland's towns are mapped the second way
almost everywhere, and more of the world every year. The footways are not drawn (roads.STYLE), and "separate" was read
as "no sidewalk": a street with its real sidewalk two metres behind a strip of grass was bare asphalt in graded ground.

Here: where each road's sidewalk really runs, measured from the footways, paths and cycleways of its own map cell
(Overture's transportation segments carry them: no download). A road is sampled every STEP_M m; at each station the
nearest path line that crosses the station's normal within REACH_M of the carriageway's edge and runs within
PARALLEL_DEG of the road counts for its side. A side with a path along COVER of the stations, at a steady distance
(SPREAD_M), has a mapped sidewalk: (the path line's distance from the road's centre line: the median; its width: the
path's mapped width, else WIDTH_M). roads._osm_sidewalks puts them on the specs (Spec.sep), roads.walk_at reads them.

One numpy pass a cell (the path points in a grid of bins, every station's 3 x 3 bins as index ranges), made with the
cell's specs and kept with them (roads.FACTS_FORMAT). Pure.
"""
import math

import numpy as np

CLASSES = ("footway", "path", "cycleway")
NOT_BESIDE = ("crosswalk", "cycle_crossing", "steps")     # subclasses: across the road, not along it
REACH_M = 8.0           # m from the carriageway's edge to the path's line at most: further, the path is the park's or
                        # the estate's behind a hedge or a parking row, and the street's sidewalk is not known
EDGE_M = 0.3            # ... and at least (a path line inside the carriageway is not its sidewalk)
STEP_M = 4.0            # m between a road's stations
END_M = 6.0             # m of each end left out (the junction: the paths turn and cross there)
ALONG_M = 1.5           # m a path point may lie off a station's normal (POINT_M apart: one always does)
POINT_M = 2.0           # m between the points a path is sampled at
PARALLEL_DEG = 25.0
COVER = 0.6             # share of a road's stations with the path beside them
SPREAD_M = 3.0          # m between the 10th and the 90th percentile of its distances at most (it runs along the road)
MIN_STATIONS = 1        # (a short piece between two junctions: its one station)
WIDTH_M = 2.0           # an unmapped width (Poland's standard is 1.8-2 m; most of the world's 1.5-2.5)
WIDTHS_M = (1.0, 6.0)   # mapped widths taken, ...
WIDTH_MAX_M = 4.0       # ... drawn this wide at most (a 5 m pavement's line is seldom its middle: the slab's outer
                        # edge stood in the pavilion behind it, Lublin, Puchacza's north end)


def _feature_ranges(feats):
    po, ro, co = feats.part_off, feats.ring_off, feats.coord_off
    idx = np.arange(len(feats))
    return np.asarray(co[ro[po[idx]]], np.int64), np.asarray(co[ro[po[idx + 1]]], np.int64)


def _segments(feats, idx, sc, origin):
    """The segments of features idx, all at once: (start points (m, 2) in metres, unit tangents (m, 2), lengths (m,),
    each segment's feature, the arc length at each segment's start counted through all of them, per feature of idx its
    (first arc length, length))."""
    a, b = _feature_ranges(feats)
    idx = np.asarray([i for i in idx if b[i] - a[i] >= 2], np.int64)
    if not len(idx):
        return None
    cnt = b[idx] - a[idx]
    first = np.cumsum(cnt) - cnt
    rows = np.repeat(a[idx] - first, cnt) + np.arange(int(cnt.sum()))
    xy = (np.asarray(feats.lonlat, np.float64)[rows] - origin) * sc
    last = np.zeros(len(rows), bool)
    last[first + cnt - 1] = True
    k = np.flatnonzero(~last)                   # (a segment from every point but each feature's last)
    d = xy[k + 1] - xy[k]
    L = np.hypot(d[:, 0], d[:, 1])
    t = d / np.maximum(L, 1e-9)[:, None]
    f = np.repeat(idx, cnt)[k]
    s0 = np.cumsum(L) - L
    nseg = cnt - 1
    fs = np.cumsum(nseg) - nseg                 # (each feature's first segment)
    start = s0[fs]
    total = np.add.reduceat(L, fs)
    return xy[k], t, L, f, s0, idx, start, total


def measure(feats, roads, paths, half, path_w=None):
    """{road: {side: (m from its centre line to the path's line, the path's width m)}} of the roads (feature indices)
    with a path (feature indices `paths`) running beside them, side +1 left / -1 right of the road's geometry. half:
    {road: its carriageway's half width m}; path_w: {path: mapped width m}. Pure."""
    out = {}
    if not len(roads) or not len(paths) or not len(feats.lonlat):
        return out
    origin = np.asarray(feats.lonlat[0], np.float64)
    la = math.radians(float(origin[1]))         # (metres a degree on the ellipsoid there: the distances are drawn)
    sc = np.array([111412.84 * math.cos(la) - 93.5 * math.cos(3 * la),
                   111132.92 - 559.82 * math.cos(2 * la) + 1.175 * math.cos(4 * la)])
    pa, ro = _segments(feats, paths, sc, origin), _segments(feats, roads, sc, origin)
    if pa is None or ro is None:
        return out
    # the paths' points: each segment's, POINT_M apart at most
    p0, ptan, pL, pfeat = pa[:4]
    cnt = np.maximum(1, np.ceil(pL / POINT_M).astype(np.int64))
    seg = np.repeat(np.arange(len(pL)), cnt)
    j = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    pp = p0[seg] + ptan[seg] * ((j + 0.5) / cnt[seg] * pL[seg])[:, None]
    pt, pf = ptan[seg], pfeat[seg]
    # the roads' stations
    r0, rtan, rL, _rf, rs0, ridx, rstart, rtotal = ro
    end = np.minimum(END_M, rtotal / 4)
    n = np.floor((rtotal - 2 * end) / STEP_M).astype(np.int64) + 1
    n[rtotal < 1e-6] = 0
    n[n < MIN_STATIONS] = 0
    if not n.sum() or not len(pp):
        return out
    feat = np.repeat(np.arange(len(ridx)), n)
    j = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    at = (rstart + end + (rtotal - 2 * end - (n - 1) * STEP_M) / 2)[feat] + STEP_M * j
    k = np.clip(np.searchsorted(rs0, at, side="right") - 1, 0, len(rs0) - 1)
    sp, st = r0[k] + rtan[k] * (at - rs0[k])[:, None], rtan[k]
    sr = ridx[feat]
    sh = np.array([float(half[int(i)]) for i in ridx])[feat]
    # the path points in bins as wide as the furthest a point may lie from its station
    B = float(sh.max()) + REACH_M + ALONG_M
    lo = np.minimum(pp.min(0), sp.min(0)) - B
    pk = np.floor((pp - lo) / B).astype(np.int64)
    sk = np.floor((sp - lo) / B).astype(np.int64)
    K = int(max(pk[:, 1].max(), sk[:, 1].max())) + 3
    key = pk[:, 0] * K + pk[:, 1]
    order = np.argsort(key, kind="stable")
    key = key[order]
    n = len(sp)
    best = np.full((n, 2), np.inf)              # per station: the nearest path's distance, left / right ...
    which = np.full((n, 2), -1, np.int64)       # ... and its feature
    cosmin = math.cos(math.radians(PARALLEL_DEG))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            k = (sk[:, 0] + dx) * K + sk[:, 1] + dy
            i0, i1 = np.searchsorted(key, k, "left"), np.searchsorted(key, k, "right")
            cnt = i1 - i0
            tot = int(cnt.sum())
            if not tot:
                continue
            si = np.repeat(np.arange(n), cnt)
            pi = order[np.repeat(i0 - (np.cumsum(cnt) - cnt), cnt) + np.arange(tot)]
            rx, ry = pp[pi, 0] - sp[si, 0], pp[pi, 1] - sp[si, 1]
            tx, ty = st[si, 0], st[si, 1]
            m = np.flatnonzero(np.abs(rx * tx + ry * ty) <= ALONG_M)        # (on the station's normal)
            si, pi, tx, ty = si[m], pi[m], tx[m], ty[m]
            lat = ry[m] * tx - rx[m] * ty                                   # (+ to the road's left)
            dist = np.abs(lat)
            m = ((np.abs(pt[pi, 0] * tx + pt[pi, 1] * ty) >= cosmin)
                 & (dist >= sh[si] + EDGE_M) & (dist <= sh[si] + REACH_M))
            si, pi, lat, dist = si[m], pi[m], lat[m], dist[m]
            col = (lat < 0).astype(np.int64)
            o = np.argsort(-dist, kind="stable")        # (the nearest written last)
            closer = dist[o] < best[si[o], col[o]]
            o = o[closer]
            best[si[o], col[o]] = dist[o]
            which[si[o], col[o]] = pf[pi[o]]
    path_w = path_w or {}
    cut = np.flatnonzero(np.diff(sr)) + 1
    for rows in np.split(np.arange(n), cut):
        i = int(sr[rows[0]])
        for col, side in ((0, 1), (1, -1)):
            d = best[rows, col]
            hit = np.isfinite(d)
            if hit.sum() < max(MIN_STATIONS, COVER * len(rows)):
                continue
            d = np.sort(d[hit])
            m = len(d)
            if d[(9 * (m - 1) + 5) // 10] - d[(m - 1) // 10] > SPREAD_M:     # (its 90th and 10th percentiles)
                continue
            ws = sorted(path_w.get(int(f), WIDTH_M) for f in which[rows, col][hit]) if path_w else (WIDTH_M,)
            out.setdefault(i, {})[side] = (float(d[m // 2]), float(ws[len(ws) // 2]))
    return out
