"""
Sokobanstein 3D
Author: Carlos Montiers Aguilera (2025)

A Sokoban game with a retro, Wolfenstein 3D-like, first-person view.

Techniques

Render pipeline (per frame)
- Clear sky/floor bands → cast walls → snapshot wall Z → draw goal pads (decals)
  → draw boxes (true 3D) → minimap → win banner.
- Per-column strict z-buffer for sprites/boxes; walls also write per-column depth.

Walls
- Classic DDA per-column ray casting over the grid; returns wall hit or map exit.
- De-fisheye: multiply raw t by cos(column-offset) (OFF_COS) before projecting.
- Height: screen-space wall slice ∝ 1/depth, scaled by WALL_HEIGHT_SCALE so you
  can see over boxes.
- Optional distance shading per column (shade_factor) applied to WALL_COLOR.

Boxes (true cubes)
- Each box is a 3D prism (width BOX_WIDTH, height BOX_HEIGHT) rendered as 5 quads
  (+X, −X, +Y, −Y, top). No face culling—z-buffer arbitrates visibility.
- Near-plane safety: all faces are transformed to camera space and Sutherland–
  Hodgman–clipped against Zc ≥ NEAR_Z before projection.
- Per-column slice rasterization: polygon → intersect with vertical scanline →
  draw [y_top, y_bot], strict z-test (with small face_bias for top only).
- Top face gets a tiny negative depth bias (TOP_FACE_EPS) to seal rim cracks.
- Raster padding ±X_PAD pixels widens ultra-thin faces to avoid dropouts.
- Lighting: simple Lambert on face normal with a fixed world LIGHT_DIR; optionally
  modulated by distance shading. Boxes on goals use BOX_ON_GOAL color.
- Draw order for cleaner pad edges: boxes are sorted far→near (by d²) but still
  z-tested per column.

Goal pads (floor decals)
- Empty goals are thin-ring "inside-only" decals: dark border quad, then slightly
  inset bright fill (GOAL_BORDER_INSET).
- Decal quads are built in world (z=0), transformed → near-clipped → projected.
- Visibility rule: decals z-test only against the snapshot of wall depth
  (pre-box) so they can appear under/around boxes.
- Then per column, decals are vertically clipped by any **nearer** box spans
  recorded that column (box_spans), with small BOX_SPAN_PAD_PIXELS inflation to
  prevent single-pixel cracks.
- Occupied goal: draw only the border decal (no fill) for visual continuity.

Shading
- Distance falloff: f = 1/(1 + SHADE_FALLOFF·depth), clamped to SHADE_MIN.
- Walls always consider distance shading when enabled; boxes can be gated via
  SPRITE_DISTANCE_SHADING. Lambert factor mixes into base colors.

Camera & projection
- Player-relative camera space: Xc=right, Yc=up, Zc=forward. Perspective:
  sx = (Xc/Zc) * INV_HALF_TAN * 0.5 + 0.5; sy = 0.5H + (H/Zc) * (0.5 − Yc).
- Columns precompute ANGLE_OFFSETS with OFF_COS/OFF_SIN for fast de-rotate rays.

Movement & pushing
- Fixed-timestep (dt=1/60). Axis-separated stepping to keep grid semantics clear.
- Single-box push with alignment tests:
  • Facing: dot (player_forward, push_axis) ≥ PUSH_FACE_DOT_MIN
  • Centering tolerance on the contacted face: PUSH_CENTER_TOL_TILES
  • Side-assist tolerance: PUSH_SIDE_ASSIST_TOL (easier near the center line)
- On successful push: move box 1 cell; leave the player slightly "flushed"
  into the vacated tile by PUSH_FLUSH_EPS to allow push-chaining.
- Undo: bounded stack (UNDO_LIMIT) stores (level, player_pos, angle).

Depth, epsilons & robustness
- NEAR_Z prevents divide-by-zero/behind-camera artifacts; all z-tests use Z_EPS.
- Top-face bias (TOP_FACE_EPS) only affects the **test** depth; recorded spans
  keep unbiased depth so pad clipping is correct.
- DECAL_DEPTH_EPS nudges decals in front of the floor to avoid tie-break z flicker.

Minimap
- Draw order mirrors world readability: walls → goals (skip if occupied) → boxes
  → player (with facing tick). Thin ring = "empty goal", box color flips to
  BOX_ON_GOAL when placed.

Level loading & start angle
- XSB parser with symbol materialization ($, ., *, @, +); floor normalized to '-'.
- Start angle auto-chosen toward the nearest visible open space (up to 2 tiles
  out), falling back to any non-wall cardinal if needed.

Controls
- Arrow keys: turn and move the player.
- U or Z: undo the last move.
- R: restart the current level.
- C: snap the player's view to the nearest cardinal angle.
- Esc: quit the game.

"""

import copy
import math
import sys

import pygame

# ===================== CONFIG =====================
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FOV = math.pi / 3
HALF_TAN = math.tan(FOV / 2)
INV_HALF_TAN = 1.0 / HALF_TAN
MAX_DEPTH = 20.0

MOVE_SPEED = 2.0
ROT_SPEED = 1.6

# Pushing feel
PUSH_FLUSH_EPS = 0.20
PUSH_FACE_DOT_MIN = 0.55
PUSH_CENTER_TOL_TILES = 0.28
PUSH_SIDE_ASSIST_TOL = 0.45

# Colors (cooler ceiling/floor; richer box tones)
CEILING_COLOR = (90, 90, 90)
FLOOR_COLOR = (125, 125, 125)
WALL_COLOR = (175, 175, 175)
BOX_COLOR = (193, 141, 101)
BOX_ON_GOAL = (248, 208, 64)
GOAL_FILL = (0, 220, 0)
GOAL_BORDER = (0, 150, 0)
GOAL_OUTLINE = (0, 140, 0)
PLY_MINIMAP = (255, 0, 0)
PLY_ON_GOAL = (0, 0, 255)

# Box dimensions (tile units)
BOX_WIDTH = 0.95
BOX_HEIGHT = 0.40

# Floor decals (pads)
GOAL_MIN_W_PX = 3
DECAL_DEPTH_EPS = 1e-3
GOAL_BORDER_INSET = 0.03  # thinner inner-fill inset => thinner ring in world
BOX_SPAN_PAD_PIXELS = 1

# Walls slightly shorter so you can see over boxes
WALL_HEIGHT_SCALE = 0.85

SHOW_MINIMAP = True
UNDO_LIMIT = 10

# Shading
ENABLE_DISTANCE_SHADING = True
SPRITE_DISTANCE_SHADING = True
SHADE_FALLOFF = 0.05
SHADE_MIN = 0.35

# Depth policy
NEAR_Z = 1e-4  # camera near plane
Z_EPS = 1e-6  # compare epsilon in z-test
TOP_FACE_EPS = 7e-5  # tiny test-only pull for the top face

# Raster padding for ultra-thin faces
X_PAD = 2  # ± pixels

# Light direction (world) for cube face shading
_LX, _LY, _LZ = (-0.55, -0.75, 0.35)
_LLEN = math.sqrt(_LX * _LX + _LY * _LY + _LZ * _LZ)
LIGHT_DIR = (_LX / _LLEN, _LY / _LLEN, _LZ / _LLEN)

# Keys
KEY_UNDO = pygame.K_u
KEY_UNDO2 = pygame.K_z
KEY_RESTART = pygame.K_r
KEY_CENTER = pygame.K_c
KEY_QUIT = pygame.K_ESCAPE

# Default fallback puzzle
DEFAULT_LEVEL_XSB = """\
--#####
###---#
#.@$--#
###-$.#
#.##$-#
#-#-.-##
#$-*$$.#
#---.--#
########
"""


# ===================== LEVEL LOADERS =====================
def parse_xsb(text: str):
    levels, current = [], []
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            if current:
                levels.append(current)
                current = []
        elif not line.lstrip().startswith(";"):
            current.append(line)
    if current:
        levels.append(current)
    return levels


def normalize_rect(level_lines, pad_char='-'):
    if not level_lines:
        return []
    width = max(len(r) for r in level_lines)
    return [list(r.ljust(width, pad_char)) for r in level_lines]


def _normalize_floor_chars(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0
    for y in range(h):
        for x in range(w):
            if grid[y][x] in ('_', ' '):
                grid[y][x] = '-'
    return grid


def materialize_symbols(grid):
    grid = _normalize_floor_chars(grid)
    player_pos = None
    goals = []
    h = len(grid)
    w = len(grid[0]) if h else 0
    for y in range(h):
        for x in range(w):
            c = grid[y][x]
            if c == '+':
                player_pos = [x + 0.5, y + 0.5]
                grid[y][x] = '.'
                goals.append((x, y))
            elif c == '*':
                grid[y][x] = '$'
                goals.append((x, y))
            elif c == '.':
                goals.append((x, y))
            elif c == '@':
                player_pos = [x + 0.5, y + 0.5]
    return grid, player_pos, goals


def load_level_from_grid(grid, preset_player=None, preset_goals=None):
    level = copy.deepcopy(grid)
    player_pos = None if preset_player is None else preset_player[:]
    goals = [] if preset_goals is None else list(preset_goals)
    start_angle = 0

    for y, row in enumerate(level):
        for x, cell in enumerate(row):
            if preset_player is None and cell == '@':
                player_pos = [x + 0.5, y + 0.5]
                row[x] = '-'
            elif cell == '@':
                row[x] = '-'
            if preset_goals is None and cell == '.':
                goals.append((x, y))

    if player_pos is None:
        raise ValueError("XSB has no player start ('@' or '+').")

    cx, cy = int(player_pos[0]), int(player_pos[1])
    cardinals = [
        ((0, -1), -math.pi / 2),
        ((1, 0), 0.0),
        ((0, 1), math.pi / 2),
        ((-1, 0), math.pi),
    ]

    def has_floor_within_two(dx, dy):
        for step in (1, 2):
            nx, ny = cx + dx * step, cy + dy * step
            if not in_bounds(nx, ny, level):
                return False
            t = level[ny][nx]
            if t == '#':
                return False
            if t in ('-', '.'):
                return True
        return False

    chosen = None
    for (dx, dy), ang in cardinals:
        if has_floor_within_two(dx, dy):
            chosen = ang
            break
    if chosen is None:
        for (dx, dy), ang in cardinals:
            nx, ny = cx + dx, cy + dy
            if in_bounds(nx, ny, level) and level[ny][nx] != '#':
                chosen = ang
                break

    start_angle = chosen if chosen is not None else 0.0
    return level, player_pos, start_angle, goals


def get_xsb_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_xsb(xsb_text):
    levels = parse_xsb(xsb_text)
    if not levels or not levels[0]:
        raise ValueError("no non-empty level found in XSB text")
    rect = normalize_rect(levels[0])
    rect, pre_player, pre_goals = materialize_symbols(rect)
    return load_level_from_grid(rect, pre_player, pre_goals)


# ===================== HELPERS =====================
def in_bounds(x, y, lvl): return 0 <= y < len(lvl) and 0 <= x < len(lvl[0])


def is_floor(c): return c in ("-", ".")


def cell_free(x, y, lvl): return in_bounds(x, y, lvl) and is_floor(lvl[y][x])


# ===================== INIT =====================
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Sokobanstein 3D")
clock = pygame.time.Clock()

if len(sys.argv) > 1:
    xsb = get_xsb_from_file(sys.argv[1])
    level, player_pos, player_angle, goals = load_xsb(xsb)
else:
    level, player_pos, player_angle, goals = load_xsb(DEFAULT_LEVEL_XSB)

initial_level = copy.deepcopy(level)
initial_player_pos = player_pos[:]
initial_player_angle = player_angle
undo_stack = []
game_won = False

# -------- precompute per-column FOV offsets --------
ANGLE_OFFSETS = [(-FOV / 2 + FOV * col / SCREEN_WIDTH)
                 for col in range(SCREEN_WIDTH)]
OFF_COS = [math.cos(a) for a in ANGLE_OFFSETS]  # de-fisheye
OFF_SIN = [math.sin(a) for a in ANGLE_OFFSETS]  # rotate rays


# ===================== SHADING =====================
def shade_factor(depth):
    f = 1.0 / (1.0 + SHADE_FALLOFF * depth)
    return max(SHADE_MIN, min(1.0, f))


def shade_color(color, factor):
    r, g, b = color
    factor = max(0.0, min(1.0, factor))
    return (int(r * factor), int(g * factor), int(b * factor))


def lambert_factor(normal_world):
    nx, ny, nz = normal_world
    lx, ly, lz = LIGHT_DIR
    dot = max(0.0, nx * lx + ny * ly + nz * lz)
    return 0.35 + 0.65 * dot


# ===================== CAMERA/PROJECTION =====================
def world_to_cam(wx, wy, wz):
    dx, dy = wx - player_pos[0], wy - player_pos[1]
    sin_a, cos_a = math.sin(player_angle), math.cos(player_angle)
    Xc = -dx * sin_a + dy * cos_a  # right
    Zc = dx * cos_a + dy * sin_a  # forward
    Yc = wz  # up
    return Xc, Yc, Zc


def project_point3(Xc, Yc, Zc):
    if Zc < NEAR_Z:
        return None
    sx = ((Xc / Zc) * INV_HALF_TAN * 0.5 + 0.5) * SCREEN_WIDTH
    sy = (SCREEN_HEIGHT * 0.5) + (SCREEN_HEIGHT / Zc) * (0.5 - Yc)
    return (sx, sy, Zc)


# ===================== RAYCAST (walls) =====================
def cast_ray_info(px, py, rx, ry, level):
    h = len(level)
    w = len(level[0]) if h else 0
    if h == 0 or w == 0:
        return MAX_DEPTH, "-", None
    INF = 1e30
    delta_x = abs(1.0 / rx) if rx != 0.0 else INF
    delta_y = abs(1.0 / ry) if ry != 0.0 else INF
    map_x, map_y = int(px), int(py)
    if rx > 0:
        step_x, side_x = 1, (map_x + 1.0 - px) * delta_x
    else:
        step_x, side_x = -1, (px - map_x) * delta_x
    if ry > 0:
        step_y, side_y = 1, (map_y + 1.0 - py) * delta_y
    else:
        step_y, side_y = -1, (py - map_y) * delta_y
    t = 0.0
    while t < MAX_DEPTH:
        if side_x < side_y:
            map_x += step_x
            t = side_x
            side_x += delta_x
        else:
            map_y += step_y
            t = side_y
            side_y += delta_y
        if map_x < 0 or map_x >= w or map_y < 0 or map_y >= h:
            return max(NEAR_Z, t), "#", (map_x, map_y)
        if level[map_y][map_x] == "#":
            return max(NEAR_Z, t), "#", (map_x, map_y)
    return MAX_DEPTH, "-", None


# ===================== NEAR-PLANE CLIP (camera space) =====================
def clip_poly_near(cam_pts, near=NEAR_Z):
    """Sutherland–Hodgman clip against plane Zc >= near."""
    if not cam_pts:
        return []
    res = []
    n = len(cam_pts)
    for i in range(n):
        S = cam_pts[i - 1]
        E = cam_pts[i]
        S_in = S[2] >= near
        E_in = E[2] >= near
        if S_in and E_in:
            res.append(E)
        elif S_in and not E_in:
            Za, Zb = S[2], E[2]
            t = (near - Za) / (Zb - Za)
            res.append((S[0] + t * (E[0] - S[0]),
                        S[1] + t * (E[1] - S[1]), near))
        elif (not S_in) and E_in:
            Za, Zb = S[2], E[2]
            t = (near - Za) / (Zb - Za)
            res.append((S[0] + t * (E[0] - S[0]),
                        S[1] + t * (E[1] - S[1]), near))
            res.append(E)
    return res


# ===================== 3D POLY RASTER =====================
def _edge_intersections_for_x_poly(x, pts):
    """Intersect convex polygon (list of (sx,sy,depth)) with vertical line x."""
    hits = []
    n = len(pts)
    for i in range(n):
        x1, y1, f1 = pts[i]
        x2, y2, f2 = pts[(i + 1) % n]
        if (x1 <= x <= x2) or (x2 <= x <= x1):
            if x2 != x1:
                t = (x - x1) / (x2 - x1)
                hits.append((y1 + t * (y2 - y1), f1 + t * (f2 - f1)))
            else:
                hits.append((y1, f1))
                hits.append((y2, f2))
    if len(hits) > 2:
        hits = sorted(hits, key=lambda p: p[0])[:2]
    return hits


def _subtract_intervals(base_start, base_end, cutters):
    segs = [(base_start, base_end)]
    for a, b in cutters:
        new = []
        for s, e in segs:
            if b <= s or a >= e:
                new.append((s, e))
            else:
                if a > s:
                    new.append((s, a))
                if b < e:
                    new.append((b, e))
        segs = new
        if not segs:
            break
    return segs


def draw_face_world_poly(world_pts, normal_world, base_color, z_buffer, box_spans, face_bias=0.0):
    """
    World verts -> camera -> near-clip -> project -> per-column slice.
    Strict z-test (stores tested z); occlusion spans record unbiased depth.
    """
    cam = [world_to_cam(wx, wy, wz) for (wx, wy, wz) in world_pts]
    cam = clip_poly_near(cam, NEAR_Z)
    if len(cam) < 3:
        return

    proj = []
    for (Xc, Yc, Zc) in cam:
        p = project_point3(Xc, Yc, Zc)
        if p is None:
            return
        proj.append(p)

    xs = [p[0] for p in proj]
    left = max(0, int(math.floor(min(xs))) - X_PAD)
    right = min(SCREEN_WIDTH - 1, int(math.ceil(max(xs))) + X_PAD)
    if right < left:
        return

    avg_depth = sum(p[2] for p in proj) / len(proj)
    brightness = lambert_factor(normal_world)
    if ENABLE_DISTANCE_SHADING and SPRITE_DISTANCE_SHADING:
        brightness *= shade_factor(avg_depth)
    color = shade_color(base_color, brightness)

    for colx in range(left, right + 1):
        hits = _edge_intersections_for_x_poly(colx + 0.5, proj)
        if len(hits) < 2:
            continue
        (y1, f1), (y2, f2) = sorted(hits, key=lambda p: p[0])
        y_top = int(max(0, min(SCREEN_HEIGHT - 1, y1)))
        y_bot = int(max(0, min(SCREEN_HEIGHT - 1, y2)))
        if y_bot <= y_top:
            continue

        forward_col = min(f1, f2)
        test_depth = forward_col - face_bias

        if test_depth < z_buffer[colx] - Z_EPS:
            pygame.draw.line(screen, color, (colx, y_top), (colx, y_bot))
            z_buffer[colx] = test_depth
            if 0 <= colx < SCREEN_WIDTH:
                box_spans[colx].append((y_top, y_bot, forward_col))


# ===================== BOX (true cube) =====================
def draw_cube(cx, cy, base_color, z_buffer, box_spans, on_goal=False):
    s = BOX_WIDTH
    h = BOX_HEIGHT
    x0 = cx - s / 2
    x1 = cx + s / 2
    y0 = cy - s / 2
    y1 = cy + s / 2

    # No face culling; let z-buffer decide.
    faces = [
        ([(x1, y0, 0.0), (x1, y1, 0.0), (x1, y1, h),
          (x1, y0, h)], (1.0, 0.0, 0.0), 0.0),  # +X
        ([(x0, y1, 0.0), (x0, y0, 0.0), (x0, y0, h),
          (x0, y1, h)], (-1.0, 0.0, 0.0), 0.0),  # -X
        ([(x1, y1, 0.0), (x0, y1, 0.0), (x0, y1, h),
          (x1, y1, h)], (0.0, 1.0, 0.0), 0.0),  # +Y
        ([(x0, y0, 0.0), (x1, y0, 0.0), (x1, y0, h),
          (x0, y0, h)], (0.0, -1.0, 0.0), 0.0),  # -Y
        ([(x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)],
         (0.0, 0.0, 1.0), TOP_FACE_EPS),  # top (biased)
    ]

    col = BOX_ON_GOAL if on_goal else base_color
    for quad, normal, bias in faces:
        draw_face_world_poly(quad, normal, col, z_buffer,
                             box_spans, face_bias=bias)


# ---------- Pads (floor decals) ----------
def _edge_intersections_for_x(x, pts):
    hits = []
    n = len(pts)
    for i in range(n):
        x1, y1, f1 = pts[i]
        x2, y2, f2 = pts[(i + 1) % n]
        if (x1 <= x <= x2) or (x2 <= x <= x1):
            if x2 != x1:
                t = (x - x1) / (x2 - x1)
                hits.append((y1 + t * (y2 - y1), f1 + t * (f2 - f1)))
            else:
                hits.append((y1, f1))
                hits.append((y2, f2))
    if len(hits) > 2:
        hits = sorted(hits, key=lambda p: p[0])[:2]
    return hits


def _goal_quad_projected(cx, cy, inset):
    """
    Build the floor-quad in world space, transform to camera space,
    clip against the near plane (Zc >= NEAR_Z), then project.
    Returns a list of (sx, sy, depth) suitable for _draw_goal_decal_zclipped_with_boxes.
    """
    half = 0.5 - inset

    # World-space floor quad (z = 0)
    world = [
        (cx - half, cy - half, 0.0),
        (cx + half, cy - half, 0.0),
        (cx + half, cy + half, 0.0),
        (cx - half, cy + half, 0.0),
    ]

    # To camera space
    cam = [world_to_cam(wx, wy, wz) for (wx, wy, wz) in world]

    # Clip against near plane so we don't drop the decal when close
    cam = clip_poly_near(cam, NEAR_Z)
    if len(cam) < 3:
        return None

    # Project to screen space, preserving depth (Zc)
    proj = []
    for (Xc, Yc, Zc) in cam:
        p = project_point3(Xc, Yc, Zc)
        if p is None:  # (shouldn't happen after clipping, but keep guard)
            return None
        proj.append(p)  # (sx, sy, depth=Zc)

    return proj


def _draw_goal_decal_zclipped_with_boxes(proj, z_walls, box_spans, color_fill):
    xs = [p[0] for p in proj]
    left = max(0, int(math.floor(min(xs))))
    right = min(SCREEN_WIDTH - 1, int(math.ceil(max(xs))))
    if right - left + 1 < GOAL_MIN_W_PX:
        return

    for colx in range(left, right + 1):
        hits = _edge_intersections_for_x(colx + 0.5, proj)
        if len(hits) < 2:
            continue
        (y1, f1), (y2, f2) = sorted(hits, key=lambda p: p[0])
        y_top = int(max(0, min(SCREEN_HEIGHT - 1, y1)))
        y_bot = int(max(0, min(SCREEN_HEIGHT - 1, y2)))
        if y_bot <= y_top:
            continue

        forward_col = min(f1, f2) - DECAL_DEPTH_EPS
        if forward_col >= z_walls[colx]:
            continue

        cutters = []
        for (bt, bb, bdepth) in box_spans[colx]:
            if bdepth + 1e-6 < forward_col:
                cutters.append((bt - BOX_SPAN_PAD_PIXELS,
                                bb + BOX_SPAN_PAD_PIXELS))
        if cutters:
            cutters.sort()
            merged = []
            cs, ce = cutters[0]
            for a, b in cutters[1:]:
                if a <= ce:
                    ce = max(ce, b)
                else:
                    merged.append((cs, ce))
                    cs, ce = a, b
            merged.append((cs, ce))
            segments = _subtract_intervals(y_top, y_bot, merged)
        else:
            segments = [(y_top, y_bot)]

        for s, e in segments:
            if e > s:
                pygame.draw.line(screen, color_fill, (colx, s), (colx, e))


def draw_goal_with_border(cx, cy, z_walls, box_spans):
    """Inside-only thin ring in world: draw dark outer, then slightly inset bright fill."""
    proj_outer = _goal_quad_projected(cx, cy, inset=0.0)
    if proj_outer is None:
        return
    _draw_goal_decal_zclipped_with_boxes(
        proj_outer, z_walls, box_spans, GOAL_BORDER)

    proj_inner = _goal_quad_projected(cx, cy, inset=GOAL_BORDER_INSET)
    if proj_inner is None:
        return
    _draw_goal_decal_zclipped_with_boxes(
        proj_inner, z_walls, box_spans, GOAL_FILL)


# ===================== DRAW SCENE =====================
def draw_scene():
    half = SCREEN_HEIGHT // 2
    screen.fill(CEILING_COLOR, (0, 0, SCREEN_WIDTH, half))
    screen.fill(FLOOR_COLOR, (0, half, SCREEN_WIDTH, half))

    z_buffer = [float("inf")] * SCREEN_WIDTH

    cos_th = math.cos(player_angle)
    sin_th = math.sin(player_angle)

    # Walls (write z-buffer per column)
    for col in range(SCREEN_WIDTH):
        rx = cos_th * OFF_COS[col] - sin_th * OFF_SIN[col]
        ry = sin_th * OFF_COS[col] + cos_th * OFF_SIN[col]
        raw_depth, hit_type, _ = cast_ray_info(
            player_pos[0], player_pos[1], rx, ry, level)
        depth = max(NEAR_Z, raw_depth * OFF_COS[col])
        if hit_type == "#":
            proj = int((SCREEN_HEIGHT * WALL_HEIGHT_SCALE) / depth)
            top, bottom = half - proj // 2, half + proj // 2
            color = shade_color(WALL_COLOR, shade_factor(
                depth)) if ENABLE_DISTANCE_SHADING else WALL_COLOR
            pygame.draw.line(screen, color, (col, top), (col, bottom))
            z_buffer[col] = depth

    wall_only_z = z_buffer[:]  # snapshot for pads

    # ==================== FIXED BLOCK START ====================
    # 1. Draw all goals first, regardless of whether they have a box on them.
    # We will use the z-buffer to determine if they are visible.
    box_spans = [[] for _ in range(SCREEN_WIDTH)]

    for gx, gy in goals:
        # If a goal is occupied, we only draw the border
        if level[gy][gx] == "$":
            # Goal under a box: draw only the border for visual continuity
            proj_outer = _goal_quad_projected(gx + 0.5, gy + 0.5, inset=0.0)
            if proj_outer is not None:
                _draw_goal_decal_zclipped_with_boxes(
                    proj_outer, wall_only_z, box_spans, GOAL_BORDER)
        else:
            # Empty goal, draw the full decal (fill + border)
            draw_goal_with_border(gx + 0.5, gy + 0.5, wall_only_z, box_spans)

    # 2. Sort boxes far->near for nicer pad edges.
    boxes = []
    for y, row in enumerate(level):
        for x, c in enumerate(row):
            if c == "$":
                d2 = (x + 0.5 - player_pos[0]) ** 2 + \
                     (y + 0.5 - player_pos[1]) ** 2
                boxes.append((d2, x, y))
    boxes.sort(reverse=True)

    # 3. Draw boxes. These will overwrite the goals in the z-buffer.
    for _, bx, by in boxes:
        on_goal = (bx, by) in goals
        draw_cube(bx + 0.5, by + 0.5, BOX_COLOR if not on_goal else BOX_ON_GOAL,
                  z_buffer, box_spans, on_goal=on_goal)
    # ==================== FIXED BLOCK END ====================

    if SHOW_MINIMAP:
        draw_minimap()

    return bool(goals and all(level[gy][gx] == "$" for gx, gy in goals))


# ===================== UI =====================
def draw_minimap():
    """Minimap with thin inside-only goal ring. Draw order: walls → goals → boxes → player.
     If a goal is occupied, we skip drawing it so only the box is visible."""
    cols = len(level[0])
    rows = len(level)
    target = 150
    cell = max(1, target // max(cols, rows))
    border = 2
    map_w = cols * cell + 2 * border
    map_h = rows * cell + 2 * border

    m = pygame.Surface((map_w, map_h), pygame.SRCALPHA)
    m.set_alpha(220)
    m.fill((0, 0, 0))
    pygame.draw.rect(m, (25, 25, 25), pygame.Rect(
        border, border, cols * cell, rows * cell))

    # 0) Walls (background)
    for y, row in enumerate(level):
        for x, c in enumerate(row):
            if c == "#":
                r = pygame.Rect(border + x * cell, border +
                                y * cell, cell, cell)
                pygame.draw.rect(m, WALL_COLOR, r, 0)

    # 1) Goals FIRST (inside-only thin ring); skip if occupied so border can't peek
    ring = max(1, cell // 12)  # thinner ring than before
    for gx, gy in goals:
        if level[gy][gx] == "$":
            continue  # occupied ⇒ only the box will be seen
        R = pygame.Rect(border + gx * cell, border + gy * cell, cell, cell)
        pygame.draw.rect(m, GOAL_OUTLINE, R, 0)  # dark ring base
        R_inner = R.inflate(-2 * ring, -2 * ring)
        if R_inner.w > 0 and R_inner.h > 0:
            pygame.draw.rect(m, GOAL_FILL, R_inner, 0)  # bright inner fill

    # 2) Boxes SECOND (slight inset so shape reads well)
    for y, row in enumerate(level):
        for x, c in enumerate(row):
            if c == "$":
                R = pygame.Rect(border + x * cell, border +
                                y * cell, cell, cell)
                inset = max(ring, cell // 8)
                RR = R.inflate(-2 * inset, -2 * inset)
                colr = BOX_ON_GOAL if (x, y) in goals else BOX_COLOR
                pygame.draw.rect(m, colr, RR, 0)

    # 3) Player
    px, py = player_pos
    cx, cy = int(border + px * cell), int(border + py * cell)
    pcol = PLY_ON_GOAL if (int(px), int(py)) in goals else PLY_MINIMAP
    pygame.draw.circle(m, pcol, (cx, cy), max(2, cell // 3))
    tipx = cx + int(math.cos(player_angle) * cell * 0.6)
    tipy = cy + int(math.sin(player_angle) * cell * 0.6)
    pygame.draw.line(m, (240, 240, 240), (cx, cy), (tipx, tipy), 2)

    screen.blit(m, (SCREEN_WIDTH - map_w - 10, 10))


def draw_win_banner():
    panel = pygame.Surface((SCREEN_WIDTH, 120), pygame.SRCALPHA)
    panel.fill((20, 20, 20, 180))
    screen.blit(panel, (0, SCREEN_HEIGHT // 2 - 60))
    font_big = pygame.font.SysFont(None, 56, bold=True)
    text = font_big.render("YOU WIN!", True, (255, 255, 0))
    sub = pygame.font.SysFont(None, 24).render(
        "Press R to restart or U/Z to undo.", True, (230, 230, 230))
    screen.blit(text, text.get_rect(
        center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 10)))
    screen.blit(sub, sub.get_rect(
        center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30)))


# ===================== PUSH & MOVEMENT =====================
def _facing_ok_for_push(dx, dy):
    fx, fy = math.cos(player_angle), math.sin(player_angle)
    if dx != 0 and dy == 0:
        return fx * (1.0 if dx > 0 else -1.0) >= PUSH_FACE_DOT_MIN
    if dy != 0 and dx == 0:
        return fy * (1.0 if dy > 0 else -1.0) >= PUSH_FACE_DOT_MIN
    return False


def _centered_on_face(px, py, bx, by, dx, dy):
    if dx != 0:
        return abs(py - (by + 0.5)) <= PUSH_CENTER_TOL_TILES
    if dy != 0:
        return abs(px - (bx + 0.5)) <= PUSH_CENTER_TOL_TILES
    return False


def _side_assist_on_face(px, py, bx, by, dx, dy):
    if dx != 0:
        return abs(py - (by + 0.5)) <= PUSH_SIDE_ASSIST_TOL
    if dy != 0:
        return abs(px - (bx + 0.5)) <= PUSH_SIDE_ASSIST_TOL
    return False


def _can_push_alignment(px, py, bx, by, dx, dy):
    return _facing_ok_for_push(dx, dy) and (
        _centered_on_face(px, py, bx, by, dx, dy) or _side_assist_on_face(px, py, bx, by, dx, dy))


def try_push(cx_from, cy_from, cx_box, cy_box, dx, dy):
    global player_pos, undo_stack, level
    if not _can_push_alignment(player_pos[0], player_pos[1], cx_box, cy_box, dx, dy):
        return False
    nx, ny = cx_box + dx, cy_box + dy
    if not cell_free(nx, ny, level):
        return False
    if len(undo_stack) >= UNDO_LIMIT:
        undo_stack.pop(0)
    undo_stack.append((copy.deepcopy(level), player_pos[:], player_angle))
    level[ny][nx] = "$"
    level[cy_box][cx_box] = '.' if (cx_box, cy_box) in goals else '-'
    player_pos = [cx_box + 0.5 + dx * PUSH_FLUSH_EPS,
                  cy_box + 0.5 + dy * PUSH_FLUSH_EPS]
    return True


def blocked_axis_move(new_coord, axis):
    global player_pos
    px, py = player_pos
    if axis == 0:
        cur_cx, cy = int(px), int(py)
        nx = new_coord
        to_cx = int(nx)
        dx = 1 if nx > px else -1
        if to_cx == cur_cx:
            player_pos[0] = nx
            return True, False, False
        if not in_bounds(to_cx, cy, level):
            return False, False, True
        target = level[cy][to_cx]
        if target == "#":
            return False, False, True
        if target == "$":
            pushed = try_push(cur_cx, cy, to_cx, cy, dx, 0)
            return (True, True, True) if pushed else (False, False, True)
        if is_floor(target):
            player_pos[0] = nx
            return True, False, False
        return False, False, True
    else:
        cx, cur_cy = int(px), int(py)
        ny = new_coord
        to_cy = int(ny)
        dy = 1 if ny > py else -1
        if to_cy == cur_cy:
            player_pos[1] = ny
            return True, False, False
        if not in_bounds(cx, to_cy, level):
            return False, False, True
        target = level[to_cy][cx]
        if target == "#":
            return False, False, True
        if target == "$":
            pushed = try_push(cx, cur_cy, cx, to_cy, 0, dy)
            return (True, True, True) if pushed else (False, False, True)
        if is_floor(target):
            player_pos[1] = ny
            return True, False, False
        return False, False, True


# ===================== CAMERA CENTER =====================
def _angle_diff(a, b): return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def center_camera():
    global player_angle
    candidates = [(-math.pi / 2, True), (0, False),
                  (math.pi / 2, True), (math.pi, False)]
    best = None
    for ang, is_vert in candidates:
        bias = 0.05 if is_vert else 0.0
        score = _angle_diff(player_angle, ang) + bias
        if best is None or score < best[0]:
            best = (score, ang)
        # keep minimal branching
    player_angle = best[1]


# ===================== MAIN LOOP =====================
def main():
    global level, player_pos, player_angle, game_won
    running = True
    while running:
        clock.tick(60)
        dt = 1.0 / 60.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (KEY_UNDO, KEY_UNDO2) and undo_stack:
                    level, player_pos, player_angle = undo_stack.pop()
                    center_camera()
                    game_won = False
                elif e.key == KEY_RESTART:
                    level[:] = [row[:] for row in initial_level]
                    player_pos[:] = initial_player_pos
                    player_angle = initial_player_angle
                    undo_stack.clear()
                    game_won = False
                elif e.key == KEY_CENTER:
                    center_camera()
                elif e.key == KEY_QUIT:
                    running = False

        move_dx = move_dy = 0.0
        if not game_won:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                player_angle -= ROT_SPEED * dt
            if keys[pygame.K_RIGHT]:
                player_angle += ROT_SPEED * dt
            if keys[pygame.K_UP]:
                move_dx += math.cos(player_angle) * MOVE_SPEED * dt
                move_dy += math.sin(player_angle) * MOVE_SPEED * dt
            if keys[pygame.K_DOWN]:
                move_dx -= math.cos(player_angle) * MOVE_SPEED * dt
                move_dy -= math.sin(player_angle) * MOVE_SPEED * dt

            if abs(move_dx) >= abs(move_dy):
                _, pushed, hard_block = blocked_axis_move(
                    player_pos[0] + move_dx, axis=0)
                if pushed or hard_block:
                    move_dy = 0.0
                if move_dy != 0.0:
                    blocked_axis_move(player_pos[1] + move_dy, axis=1)
            else:
                _, pushed, hard_block = blocked_axis_move(
                    player_pos[1] + move_dy, axis=1)
                if pushed or hard_block:
                    move_dx = 0.0
                if move_dx != 0.0:
                    blocked_axis_move(player_pos[0] + move_dx, axis=0)

        did_win = draw_scene()
        game_won = game_won or did_win
        if game_won:
            draw_win_banner()
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
