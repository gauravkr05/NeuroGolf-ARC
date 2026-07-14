"""Vocabulary v2: ~42 search operations = 20 probed unary DSL primitives
+ 22 macros (binary self-combos, thirds, shifts, dynamic-color ops).
Each entry: name -> (callable(grid, *args) -> grid, argspec)
argspec: "g" no args | "gc" one color | "gcc" two colors | "gk" factor
"""
import sys, inspect, random
sys.path.insert(0, "/home/claude/arc-dsl")
import dsl

# ---- probe the original unary prims (same logic as gen_synth2) ----
def is_grid(x):
    return (isinstance(x, tuple) and 1 <= len(x) <= 30 and
            all(isinstance(r, tuple) and len(r) == len(x[0]) and len(r) >= 1
                and all(isinstance(v, int) and not isinstance(v, bool)
                        and 0 <= v <= 9 for v in r) for r in x))

BANNED = {"either", "toindices", "totuple", "asindices", "double",
          "multiply", "identity", "canvas"}
FACTOR_PRIMS = {"upscale", "downscale", "hupscale", "vupscale"}

_rng = random.Random(7)
_probe = tuple(tuple(_rng.randint(0, 9) for _ in range(4)) for _ in range(5))

VOCAB = {}
for name, fn in inspect.getmembers(dsl, inspect.isfunction):
    if name in BANNED:
        continue
    if name in FACTOR_PRIMS:
        VOCAB[name] = (fn, "gk")
        continue
    for spec, args in [("g", (_probe,)), ("gc", (_probe, 3)),
                       ("gcc", (_probe, 3, 5))]:
        try:
            if is_grid(fn(*args)):
                VOCAB[name] = (fn, spec)
                break
        except Exception:
            pass

# ---- macros ----
def _bg(g):
    return dsl.mostcolor(g)

def hconcat_dup(g):        return dsl.hconcat(g, g)
def vconcat_dup(g):        return dsl.vconcat(g, g)
def hconcat_vmirror_r(g):  return dsl.hconcat(g, dsl.vmirror(g))
def hconcat_vmirror_l(g):  return dsl.hconcat(dsl.vmirror(g), g)
def vconcat_hmirror_b(g):  return dsl.vconcat(g, dsl.hmirror(g))
def vconcat_hmirror_t(g):  return dsl.vconcat(dsl.hmirror(g), g)
def hconcat_rot180(g):     return dsl.hconcat(g, dsl.rot180(g))
def vconcat_rot180(g):     return dsl.vconcat(g, dsl.rot180(g))

def left_third(g):
    if len(g[0]) % 3: raise ValueError
    return dsl.hsplit(g, 3)[0]
def right_third(g):
    if len(g[0]) % 3: raise ValueError
    return dsl.hsplit(g, 3)[2]
def top_third(g):
    if len(g) % 3: raise ValueError
    return dsl.vsplit(g, 3)[0]
def bottom_third(g):
    if len(g) % 3: raise ValueError
    return dsl.vsplit(g, 3)[2]

def shift_down(g):
    w = len(g[0]); return (tuple(0 for _ in range(w)),) + g[:-1]
def shift_up(g):
    w = len(g[0]); return g[1:] + (tuple(0 for _ in range(w)),)
def shift_right(g):
    return tuple((0,) + r[:-1] for r in g)
def shift_left(g):
    return tuple(r[1:] + (0,) for r in g)

def swap_most_least(g):
    a, b = dsl.mostcolor(g), dsl.leastcolor(g)
    if a == b: raise ValueError
    return dsl.switch(g, a, b)
def replace_least(g, c):
    return dsl.replace(g, dsl.leastcolor(g), c)
def replace_most(g, c):
    return dsl.replace(g, dsl.mostcolor(g), c)
def keep_only(g, c):
    if all(v != c for r in g for v in r): raise ValueError
    return tuple(tuple(v if v == c else 0 for v in r) for r in g)

def pad_border(g, c):
    w = len(g[0]) + 2
    row = tuple(c for _ in range(w))
    return (row,) + tuple((c,) + r + (c,) for r in g) + (row,)

def outline_box(g, c):
    bg = _bg(g)
    cells = [(i, j) for i, r in enumerate(g) for j, v in enumerate(r)
             if v != bg]
    if not cells: raise ValueError
    i0 = min(i for i, _ in cells); i1 = max(i for i, _ in cells)
    j0 = min(j for _, j in cells); j1 = max(j for _, j in cells)
    out = [list(r) for r in g]
    for j in range(j0, j1 + 1):
        out[i0][j] = c; out[i1][j] = c
    for i in range(i0, i1 + 1):
        out[i][j0] = c; out[i][j1] = c
    return tuple(tuple(r) for r in out)

MACROS = {
    "hconcat_dup": (hconcat_dup, "g"), "vconcat_dup": (vconcat_dup, "g"),
    "hconcat_vmirror_r": (hconcat_vmirror_r, "g"),
    "hconcat_vmirror_l": (hconcat_vmirror_l, "g"),
    "vconcat_hmirror_b": (vconcat_hmirror_b, "g"),
    "vconcat_hmirror_t": (vconcat_hmirror_t, "g"),
    "hconcat_rot180": (hconcat_rot180, "g"),
    "vconcat_rot180": (vconcat_rot180, "g"),
    "left_third": (left_third, "g"), "right_third": (right_third, "g"),
    "top_third": (top_third, "g"), "bottom_third": (bottom_third, "g"),
    "shift_down": (shift_down, "g"), "shift_up": (shift_up, "g"),
    "shift_right": (shift_right, "g"), "shift_left": (shift_left, "g"),
    "swap_most_least": (swap_most_least, "g"),
    "replace_least": (replace_least, "gc"),
    "replace_most": (replace_most, "gc"),
    "keep_only": (keep_only, "gc"),
    "pad_border": (pad_border, "gc"),
    "outline_box": (outline_box, "gc"),
}
VOCAB.update(MACROS)
NAMES = sorted(VOCAB)

def apply_step(name, args, g):
    return VOCAB[name][0](g, *args)

if __name__ == "__main__":
    print(f"{len(NAMES)} operations:")
    print(NAMES)

# ================= v2 extension: 42 -> 62 ops =================
def _cellwise(a, b, f):
    if len(a) != len(b) or len(a[0]) != len(b[0]): raise ValueError
    return tuple(tuple(f(x, y) for x, y in zip(ra, rb))
                 for ra, rb in zip(a, b))

def _lr(g): return dsl.lefthalf(g), dsl.righthalf(g)
def _tb(g): return dsl.tophalf(g), dsl.bottomhalf(g)

def lr_and(g, c):
    a, b = _lr(g); return _cellwise(a, b, lambda x, y: c if x and y else 0)
def lr_or(g, c):
    a, b = _lr(g); return _cellwise(a, b, lambda x, y: c if x or y else 0)
def lr_xor(g, c):
    a, b = _lr(g); return _cellwise(a, b, lambda x, y: c if bool(x) != bool(y) else 0)
def lr_diff(g, c):
    a, b = _lr(g); return _cellwise(a, b, lambda x, y: c if x and not y else 0)
def tb_and(g, c):
    a, b = _tb(g); return _cellwise(a, b, lambda x, y: c if x and y else 0)
def tb_or(g, c):
    a, b = _tb(g); return _cellwise(a, b, lambda x, y: c if x or y else 0)
def tb_xor(g, c):
    a, b = _tb(g); return _cellwise(a, b, lambda x, y: c if bool(x) != bool(y) else 0)
def tb_diff(g, c):
    a, b = _tb(g); return _cellwise(a, b, lambda x, y: c if x and not y else 0)

RING8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
ORTH4 = [(-1,0),(1,0),(0,-1),(0,1)]
DIAG4 = [(-1,-1),(-1,1),(1,-1),(1,1)]
def _paint(g, src, dst, offs):
    h, w = len(g), len(g[0]); out = [list(r) for r in g]; hit = False
    for i, r in enumerate(g):
        for j, v in enumerate(r):
            if v == src:
                hit = True
                for di, dj in offs:
                    y, x = i + di, j + dj
                    if 0 <= y < h and 0 <= x < w and g[y][x] == 0:
                        out[y][x] = dst
    if not hit: raise ValueError
    return tuple(tuple(r) for r in out)
def paint_ring(g, s, d): return _paint(g, s, d, RING8)
def paint_orth(g, s, d): return _paint(g, s, d, ORTH4)
def paint_diag(g, s, d): return _paint(g, s, d, DIAG4)

def fill_bbox_delta(g, target, fillc):
    cells = [(i, j) for i, r in enumerate(g) for j, v in enumerate(r)
             if v == target]
    if not cells: raise ValueError
    i0, i1 = min(i for i,_ in cells), max(i for i,_ in cells)
    j0, j1 = min(j for _,j in cells), max(j for _,j in cells)
    out = [list(r) for r in g]
    for i in range(i0, i1+1):
        for j in range(j0, j1+1):
            if g[i][j] != target:
                out[i][j] = fillc
    return tuple(tuple(r) for r in out)

def quad_symmetrize(g):
    top = dsl.hconcat(g, dsl.vmirror(g))
    return dsl.vconcat(top, dsl.hmirror(top))
def htile(g, k):
    out = g
    for _ in range(k - 1): out = dsl.hconcat(out, g)
    return out
def vtile(g, k):
    out = g
    for _ in range(k - 1): out = dsl.vconcat(out, g)
    return out

def recolor_nonzero(g, c):
    return tuple(tuple(c if v else 0 for v in r) for r in g)
def crop_content(g):
    bg = dsl.mostcolor(g)
    cells = [(i, j) for i, r in enumerate(g) for j, v in enumerate(r) if v != bg]
    if not cells: raise ValueError
    i0, i1 = min(i for i,_ in cells), max(i for i,_ in cells)
    j0, j1 = min(j for _,j in cells), max(j for _,j in cells)
    return tuple(r[j0:j1+1] for r in g[i0:i1+1])
def subgrid_color(g, c):
    cells = [(i, j) for i, r in enumerate(g) for j, v in enumerate(r) if v == c]
    if not cells: raise ValueError
    i0, i1 = min(i for i,_ in cells), max(i for i,_ in cells)
    j0, j1 = min(j for _,j in cells), max(j for _,j in cells)
    return tuple(r[j0:j1+1] for r in g[i0:i1+1])
def vdedupe(g):
    return dsl.dmirror(dsl.dedupe(dsl.dmirror(g)))
def gravity_down(g):
    h, w = len(g), len(g[0])
    cols = []
    for j in range(w):
        vals = [g[i][j] for i in range(h) if g[i][j]]
        cols.append([0]*(h-len(vals)) + vals)
    return tuple(tuple(cols[j][i] for j in range(w)) for i in range(h))

V2 = {
 "lr_and": (lr_and,"gc"), "lr_or": (lr_or,"gc"), "lr_xor": (lr_xor,"gc"),
 "lr_diff": (lr_diff,"gc"), "tb_and": (tb_and,"gc"), "tb_or": (tb_or,"gc"),
 "tb_xor": (tb_xor,"gc"), "tb_diff": (tb_diff,"gc"),
 "paint_ring": (paint_ring,"gcc"), "paint_orth": (paint_orth,"gcc"),
 "paint_diag": (paint_diag,"gcc"), "fill_bbox_delta": (fill_bbox_delta,"gcc"),
 "quad_symmetrize": (quad_symmetrize,"g"), "htile": (htile,"gk"),
 "vtile": (vtile,"gk"), "recolor_nonzero": (recolor_nonzero,"gc"),
 "crop_content": (crop_content,"g"), "subgrid_color": (subgrid_color,"gc"),
 "vdedupe": (vdedupe,"g"), "gravity_down": (gravity_down,"g"),
}
VOCAB.update(V2)
NAMES = sorted(VOCAB)
