"""The meshes of GUGiK's LoD2 building models (pl_lod2.Model on a Spec: spec.lod2), drawn with the city's materials.

Walls: each wall polygon of the model (vertical, planar) in its own plane (s along the wall, seen from outside to the
right; h its height over the building's ground), cut into the storeys of the window sheet: the plinth under the floors
(the ground's slope under the footprint, plain), then one band of the facade sheet a storey (spec.floor_h, from plan()
on the model's eave height) up to the wall's top, gables and steps included; a shopfront ground floor where plan()
says so; whole bays along each wall as buildings._walls. A plain wall material (spec.wall: a house's plaster; spec.plain:
sheds, barns) is tiled at SIDING_UV_M, a house's windows laid over it where a whole window fits in the polygon.
Roofs: each roof polygon as modelled, its courtyards kept (holes), the roof material tiled at ROOF_UV_M.
Collision: the model's walls (to FOUNDATION_M under the lowest ground) and roofs, uncut.
"""
import math

import numpy as np

import buildings as B
import pl_lod2

GROUND_EPS = 0.05           # m: a wall vertex this close to the model's ground is on it (lowered under the game's)
AREA_TOL = 0.02             # a fast triangulation that lost more than this share of the area is done again by _earclip


# ------------------------------------------------------------------------------------------------ 2D helpers

def _area2(p):
    x, y = p[:, 0], p[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _tri_area_sum(p, tris):
    if not len(tris):
        return 0.0
    a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    return float(np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum())


def _earclip(p):
    """Ear clipping of the counter-clockwise ring p (n, 2) that may touch itself (a polygon's holes bridged into it):
    a vertex at the same place as one of a candidate ear's corners does not block it. (k, 3) indices."""
    n = len(p)
    idx = list(range(n))
    out = []
    pts = [tuple(map(float, q)) for q in p]

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    guard = 0
    while len(idx) > 3 and guard < 4 * n * n:
        guard += 1
        m = len(idx)
        clipped = False
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 1e-12:
                continue
            ok = True
            for j in idx:
                q = pts[j]
                if j in (i0, i1, i2) or q == a or q == b or q == c:
                    continue
                if cross(a, b, q) >= -1e-12 and cross(b, c, q) >= -1e-12 and cross(c, a, q) >= -1e-12:
                    ok = False
                    break
            if ok:
                out.append((i0, i1, i2))
                del idx[k]
                clipped = True
                break
        if not clipped:         # what is left has no ear (degenerate, self-crossing): drop a flat vertex, or stop
            flat = [k for k in range(m) if abs(cross(pts[idx[k - 1]], pts[idx[k]], pts[idx[(k + 1) % m]])) <= 1e-9]
            if not flat:
                break
            del idx[flat[0]]
    if len(idx) == 3 and cross(pts[idx[0]], pts[idx[1]], pts[idx[2]]) > 1e-12:
        out.append(tuple(idx))
    return np.array(out, np.int64).reshape(-1, 3)


def triangulate(p):
    """Triangles (k, 3) of the ring p (n, 2), counter-clockwise or not (the indices' order: counter-clockwise)."""
    p = np.asarray(p, np.float64)
    if len(p) < 3:
        return np.zeros((0, 3), np.int64)
    a = _area2(p)
    if a < 0:
        n = len(p)
        t = triangulate(p[::-1])
        return (n - 1 - t)[:, ::-1] if len(t) else t
    if a <= 1e-9:
        return np.zeros((0, 3), np.int64)
    t = B.triangulate(p) if len(p) > 3 else np.array([[0, 1, 2]], np.int64)
    if _tri_area_sum(p, t) < (1 - AREA_TOL) * a:
        t2 = _earclip(p)
        if _tri_area_sum(p, t2) > _tri_area_sum(p, t):
            t = t2
    return t


def bridge(outer, holes):
    """One ring (k, 2) of a counter-clockwise outer ring and its holes, each joined to it by a cut (a pair of
    coincident edges) from its right-most vertex to the nearest outer vertex the cut does not cross anything to."""
    ring = np.asarray(outer, np.float64)
    if _area2(ring) < 0:
        ring = ring[::-1]
    hs = []
    for h in holes:
        h = np.asarray(h, np.float64)
        if len(h) >= 3 and abs(_area2(h)) > 1e-6:
            hs.append(h[::-1] if _area2(h) > 0 else h)         # holes clockwise
    hs.sort(key=lambda h: -h[:, 0].max())
    for h in hs:
        j = int(np.argmax(h[:, 0]))
        hp = h[j]
        segs = [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
        for g in hs:
            segs += [(g[i], g[(i + 1) % len(g)]) for i in range(len(g))]
        d = np.hypot(*(ring - hp).T)
        best = None
        for i in np.argsort(d):
            q = ring[i]
            if not any(_cross_proper(hp, q, s0, s1) for s0, s1 in segs):
                best = int(i)
                break
        if best is None:
            continue                    # (a hole that cannot be reached: left out, its area roofed)
        hh = np.r_[h[j:], h[:j + 1]]
        ring = np.r_[ring[:best + 1], hh, ring[best:]]
    return ring


def _cross_proper(a, b, c, d):
    """Whether segments ab and cd cross at a point inside both (touching at an end does not count)."""
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else -1)
    if any(np.allclose(x, y) for x in (a, b) for y in (c, d)):
        return False
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def _clip_s(poly, lo, hi):
    """The part of the 2D polygon poly [(s, h)] with lo <= s <= hi."""
    got = _clip([(h, s) for s, h in poly], lo, hi)
    return [(s, h) for h, s in got]


def _clip(poly, lo, hi):
    """The part of the 2D polygon poly [(s, h)] with lo <= h <= hi (Sutherland-Hodgman, two half-planes)."""
    def cut(pts, keep, z):
        out = []
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            ia, ib = keep(a[1]), keep(b[1])
            if ia:
                out.append(a)
            if ia != ib and a[1] != b[1]:
                t = (z - a[1]) / (b[1] - a[1])
                out.append((a[0] + t * (b[0] - a[0]), z))
        return out
    pts = cut(poly, lambda h: h >= lo, lo)
    if len(pts) >= 3:
        pts = cut(pts, lambda h: h <= hi, hi)
    return pts if len(pts) >= 3 else []


# ------------------------------------------------------------------------------------------------ walls

def _newell(v):
    n = np.zeros(3)
    for i in range(len(v)):
        a, b = v[i], v[(i + 1) % len(v)]
        n += ((a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1]))
    return n


def _band_v(k):
    eps = 1.5 / 1024.0
    return 1.0 - (k + 1) / B.BANDS + eps, 1.0 - k / B.BANDS - eps


class _Wall:
    """A wall polygon in its plane: s (to the right seen from outside), h; its outward normal and origin."""
    __slots__ = ("pts", "r", "n", "o", "smin", "smax", "hmin", "hmax")

    def __init__(self, v):
        nrm = _newell(v)
        nh = math.hypot(nrm[0], nrm[1])
        if nh < 1e-9:
            self.pts = None
            return
        n = np.array([nrm[0] / nh, nrm[1] / nh])
        r = np.array([-n[1], n[0]])                 # (right, seen from outside: cross(-n, up))
        o = v[0, :2]
        s = (v[:, :2] - o) @ r
        self.pts = [(float(a), float(b)) for a, b in zip(s, v[:, 2])]
        self.r, self.n, self.o = r, n, o
        self.smin, self.smax = float(s.min()), float(s.max())
        self.hmin, self.hmax = float(v[:, 2].min()), float(v[:, 2].max())

    def xyz(self, s, h, base, out=0.0):
        p = self.o + self.r * s + self.n * out
        return float(p[0]), float(p[1]), base + h


def _emit(m, wall, pts, uv_of, base, mat):
    """The 2D polygon pts [(s, h)] of a wall, triangulated, uvs by uv_of(s, h)."""
    p = np.asarray(pts, np.float64)
    t = triangulate(p)
    if not len(t):
        return
    P = [wall.xyz(s, h, base) for s, h in pts]
    UV = [uv_of(s, h) for s, h in pts]
    for a, b, c in t:
        m.tri(P[a], P[b], P[c], UV[a], UV[b], UV[c], mat)


def _is_rect(pts):
    """(s0, s1, h0, h1) of a polygon that is an upright rectangle in its plane, else None."""
    if len(pts) != 4:
        return None
    ss = sorted({round(s, 4) for s, _ in pts})
    hs = sorted({round(h, 4) for _, h in pts})
    if len(ss) != 2 or len(hs) != 2:
        return None
    return ss[0], ss[1], hs[0], hs[1]


def _walls(m, spec, model, base, z0, mat_idx, lite):
    """Every wall polygon of the model (see the module docstring). base: the floors' ground (absolute); z0: the walls'
    bottom (absolute, under the lowest ground)."""
    fac = mat_idx[spec.facade]
    shop_mat = mat_idx[B.SHOP]
    white = mat_idx["bne_bld_fac_white"]
    plain_tile = bool(spec.wall) or bool(spec.plain)
    band = spec.band or (0, 1)
    bk, bpt = int(band[0]), int(band[1])
    fh = max(float(spec.floor_h or B.FLOOR_M), 1.0)
    bay = B.TILE_PX_M / bpt
    plinth_v = _band_v(bk)
    sk = B.SIDING_UV_M
    good = list(band[2]) if len(band) > 2 and not lite else []
    seed = float(spec.seed) if isinstance(spec.seed, float) else 0.0
    hb = z0 - base                                  # the walls' bottom, over the floors' ground (negative)
    for wi, (kind, rings) in enumerate(model.polys):
        if kind != pl_lod2.WALL:
            continue
        s_, e_ = rings[0]
        v = np.array(model.verts[s_:e_], np.float64)
        v[v[:, 2] <= GROUND_EPS, 2] = hb            # (walls standing on the model's ground reach under the game's)
        w = _Wall(v)
        if w.pts is None or w.smax - w.smin < 0.05:
            continue
        L = w.smax - w.smin
        if plain_tile:
            _emit(m, w, w.pts, lambda s, h: (s / sk, h / sk), base, fac)
            if spec.wall and spec.windows is not False and not lite:
                _windows(m, spec, w, base, white)
            continue
        nb = int(round(L / bay))
        uscale = (nb / bpt) / L if nb >= 1 else 1.0 / B.TILE_PX_M
        rect = _is_rect(w.pts)
        top = w.hmax
        # the plinth: from the walls' bottom to the floors' ground, the band's plain strip (one colour)
        cuts = []
        if w.hmin < 0.0:
            cuts.append((w.hmin, 0.0, None, 0))
        f = 0
        while f * fh < top - 1e-3:
            cuts.append((f * fh, (f + 1) * fh, f, 0))
            f += 1
        for lo, hi, fl, _ in cuts:
            if fl is None:
                uc = 1.0 / bpt
                v0, v1 = plinth_v
                uv = (lambda s, h, uc=uc, v0=v0: (uc, v0 + 0.02))
                mat = fac
            else:
                shop = fl == 0 and spec.shop
                k = B.BAND_SHOP[0] if shop else bk
                v0, v1 = _band_v(k)
                if shop:
                    nbs = max(int(round(L / (B.TILE_PX_M / B.BAND_SHOP[1]))), 1)
                    us = (nbs / B.BAND_SHOP[1]) / L
                else:
                    us = uscale
                uv = (lambda s, h, lo=lo, us=us, v0=v0, v1=v1:
                      ((s - w.smin) * us, v0 + (h - lo) / fh * (v1 - v0)))
                mat = shop_mat if shop else fac
            if rect is not None:
                s0, s1, h0, h1 = rect
                a, b = max(lo, h0), min(hi, h1)
                if b - a < 1e-3:
                    continue
                pts = [(s0, a), (s1, a), (s1, b), (s0, b)]
            else:
                pts = _clip(w.pts, lo, hi)
                if not pts:
                    continue
            if fl is not None and good and nb >= 1 and mat is fac:
                # one bay at a time, each a good bay of the band (buildings._walls: the sheet's black "open" windows
                # left out), hashed as the walls' kernel hashes them
                bw = L / nb
                for k_ in range(nb):
                    g = good[int(B._hash01(seed, wi, fl, k_) * len(good))]
                    sa, sb = w.smin + k_ * bw, w.smin + (k_ + 1) * bw
                    if rect is not None:
                        part = [(sa, pts[0][1]), (sb, pts[0][1]), (sb, pts[2][1]), (sa, pts[2][1])]
                    else:
                        part = _clip_s(pts, sa, sb)
                        if not part:
                            continue
                    _emit(m, w, part, (lambda s, h, lo=lo, sa=sa, g=g, v0=v0, v1=v1:
                                       ((g + (s - sa) / bw) / bpt, v0 + (h - lo) / fh * (v1 - v0))), base, mat)
                continue
            _emit(m, w, pts, uv, base, mat)


def _windows(m, spec, w, base, white):
    """A house's windows laid over its plain wall (buildings._walls' plain wall with windows), where one fits whole."""
    from polygon import Path
    L = w.smax - w.smin
    nb = int(L // B.HOUSE_BAY_M)
    if nb < 1 or L < 2.0:
        return
    path = Path(np.asarray(w.pts, np.float64))
    u0, u1, v0, v1 = B.WIN_UV
    fh = max(float(spec.floor_h or B.FLOOR_M), 1.0)
    f = 0
    while (f + 1) * fh <= w.hmax + 1e-3 and f < 60:
        zf = f * fh + B.WIN_SILL
        for k in range(nb):
            if f == 0 and nb % 2 == 1 and k == nb // 2:
                continue                # (the door's bay)
            sc = w.smin + L * (k + 0.5) / nb
            s0, s1 = sc - B.WIN_W / 2, sc + B.WIN_W / 2
            corners = np.array([(s0, zf), (s1, zf), (s1, zf + B.WIN_H), (s0, zf + B.WIN_H)])
            if not path.contains_points(corners).all():
                continue
            P = [w.xyz(s, h, base, B.WIN_OUT_M) for s, h in corners]
            m.quad(P[0], P[1], P[2], P[3], (u0, v0), (u1, v0), (u1, v1), (u0, v1), white)
        f += 1


# ------------------------------------------------------------------------------------------------ roofs

def _roofs(m, model, base, mat):
    for kind, rings in model.polys:
        if kind != pl_lod2.ROOF:
            continue
        outer = model.verts[rings[0][0]:rings[0][1]]
        holes = [model.verts[s:e] for s, e in rings[1:]]
        if holes:
            ring = bridge(outer[:, :2], [h[:, :2] for h in holes])
            # (the bridged ring's heights: from the polygon's plane, through its outer ring)
            z = _plane_z(np.r_[outer, *holes], ring)
            pts = np.c_[ring, z]
        else:
            pts = np.asarray(outer, np.float64)
        if _area2(pts[:, :2]) < 0:
            continue                    # (a roof polygon facing down: an overhang's underside, not drawn)
        t = triangulate(pts[:, :2])
        if not len(t):
            continue
        P = [(float(x), float(y), base + float(z)) for x, y, z in pts]
        UV = [(float(x) / B.ROOF_UV_M, float(y) / B.ROOF_UV_M) for x, y, _ in pts]
        for a, b, c in t:
            m.tri(P[a], P[b], P[c], UV[a], UV[b], UV[c], mat)


def _plane_z(v, xy):
    """Heights at xy (k, 2) of the plane through the points v (n, 3) (least squares)."""
    A = np.c_[v[:, 0], v[:, 1], np.ones(len(v))]
    coef, *_ = np.linalg.lstsq(A, v[:, 2], rcond=None)
    return xy[:, 0] * coef[0] + xy[:, 1] * coef[1] + coef[2]


# ------------------------------------------------------------------------------------------------ the building

def draw(m, spec, gmin, gmax, mat_idx, lite=False, collision=False, tok=None):
    """Append the model of spec (spec.lod2: a pl_lod2.Model in the tile's frame) to the Mesh m. collision: plain
    walls and roofs (token tok)."""
    model = spec.lod2
    base = B.base_z(spec, gmin, gmax)
    if collision:
        hb = gmin - B.FOUNDATION_M - base
        zero = (0.0, 0.0)
        for kind, rings in model.polys:
            s_, e_ = rings[0]
            v = np.array(model.verts[s_:e_], np.float64)
            if kind == pl_lod2.WALL:
                v[v[:, 2] <= GROUND_EPS, 2] = hb
                w = _Wall(v)
                if w.pts is None:
                    continue
                _emit(m, w, w.pts, lambda s, h: zero, base, tok)
            else:
                if _area2(v[:, :2]) <= 0:
                    continue
                t = triangulate(v[:, :2])
                P = [(float(x), float(y), base + float(z)) for x, y, z in v]
                for a, b, c in t:
                    m.tri(P[a], P[b], P[c], zero, zero, zero, tok)
        return
    z0 = gmin - (B.SKIRT_LITE_M if lite else B.SKIRT_M)
    _walls(m, spec, model, base, z0, mat_idx, lite)
    _roofs(m, model, base, mat_idx[spec.roof_mat])
