"""
Utility functions for PROJECT VERTEX
"""
import json
import math
import os
from datetime import datetime
import pygame


# ---------------------------------------------------------------------------
# Blueprint (save / load)
# ---------------------------------------------------------------------------

def save_blueprint(shape, rot_x, rot_y, zoom, color=None, filename="blueprint.json"):
    """Save current state to a blueprint file."""
    data = {
        "project_name": "VERTEX_V2",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "shape": shape,
        "rotation": {"x": round(rot_x, 2), "y": round(rot_y, 2)},
        "zoom_level": round(zoom, 2),
        "color": color if color else [0.0, 1.0, 1.0],
    }
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f">> 💾 SAVED BLUEPRINT: {filename}")
        return True
    except Exception as e:
        print(f">> ❌ ERROR SAVING: {e}")
        return False


def load_blueprint(filename="blueprint.json"):
    """Load state from a blueprint file."""
    if not os.path.exists(filename):
        print(f">> ⚠️  No blueprint found at: {filename}")
        return None
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        ts = data.get("timestamp", "unknown")
        print(f">> 📂 LOADED BLUEPRINT: {filename}  (saved {ts})")
        return data
    except Exception as e:
        print(f">> ❌ ERROR LOADING: {e}")
        return None


def screenshot(surface=None, directory=".") -> str:
    """Save the current OpenGL framebuffer to a timestamped PNG.

    Returns the filepath written (empty string on error).
    """
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"vertex_{ts}.png"
    path = os.path.join(directory, name)
    try:
        from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
        w, h = pygame.display.get_surface().get_size()
        raw  = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
        surf = pygame.image.fromstring(raw, (w, h), "RGB")
        surf = pygame.transform.flip(surf, False, True)   # OpenGL Y-flip
        pygame.image.save(surf, path)
    except Exception as e:
        print(f">> ⚠️  Screenshot error: {e}")
        return ""
    return path


# ---------------------------------------------------------------------------
# OBJ Export — per-shape geometry builders
# ---------------------------------------------------------------------------

def _build_cube(size=1.0):
    s = size
    verts = [
        ( s, -s, -s), ( s,  s, -s), (-s,  s, -s), (-s, -s, -s),
        ( s, -s,  s), ( s,  s,  s), (-s, -s,  s), (-s,  s,  s),
    ]
    # OBJ face indices are 1-based; each tuple is one quad split into 2 triangles
    faces = [
        (1,2,3,4), (5,8,7,6), (1,5,6,2),
        (3,7,8,4), (1,4,8,5), (2,6,7,3),
    ]
    # Split quads → triangles
    tris = []
    for a, b, c, d in faces:
        tris.append((a, b, c))
        tris.append((a, c, d))
    return verts, tris


def _build_pyramid(size=1.0):
    s = size
    verts = [
        (0, s, 0),
        (-s, -s,  s), ( s, -s,  s),
        ( s, -s, -s), (-s, -s, -s),
    ]
    # Sides (triangles) + base (quad split into 2 triangles), 1-based
    tris = [
        (1,2,3), (1,3,4), (1,4,5), (1,5,2),
        (2,3,4), (2,4,5),  # base
    ]
    return verts, tris


def _build_sphere(radius=1.0, slices=24, stacks=24):
    verts = []
    for j in range(stacks + 1):
        phi = math.pi * j / stacks
        for i in range(slices):
            theta = 2 * math.pi * i / slices
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)
            verts.append((x, y, z))

    tris = []
    for j in range(stacks):
        for i in range(slices):
            a = j * slices + i + 1          # 1-based
            b = j * slices + (i + 1) % slices + 1
            c = (j + 1) * slices + i + 1
            d = (j + 1) * slices + (i + 1) % slices + 1
            tris.append((a, b, d))
            tris.append((a, d, c))
    return verts, tris


def _build_cylinder(radius=1.0, height=2.0, segments=24):
    half_h = height / 2
    verts = []
    # Bottom ring then top ring
    for y in [-half_h, half_h]:
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            verts.append((radius * math.cos(angle), y, radius * math.sin(angle)))
    # Centres
    verts.append((0, -half_h, 0))   # index: 2*segments+1 (1-based)
    verts.append((0,  half_h, 0))   # index: 2*segments+2

    n = segments
    tris = []
    # Side quads
    for i in range(n):
        b = i + 1
        t = n + i + 1
        bn = (i + 1) % n + 1
        tn = n + (i + 1) % n + 1
        tris.append((b, bn, tn))
        tris.append((b, tn, t))
    # Bottom cap
    bot_center = 2 * n + 1
    for i in range(n):
        tris.append((bot_center, (i + 1) % n + 1, i + 1))
    # Top cap
    top_center = 2 * n + 2
    for i in range(n):
        tris.append((top_center, n + i + 1, n + (i + 1) % n + 1))
    return verts, tris


def _build_torus(inner_r=0.3, outer_r=1.0, segments=24, rings=24):
    verts = []
    for i in range(rings):
        u = 2 * math.pi * i / rings
        for j in range(segments):
            v = 2 * math.pi * j / segments
            r = outer_r + inner_r * math.cos(v)
            verts.append((r * math.cos(u), inner_r * math.sin(v), r * math.sin(u)))

    tris = []
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j + 1
            b = i * segments + (j + 1) % segments + 1
            c = ((i + 1) % rings) * segments + j + 1
            d = ((i + 1) % rings) * segments + (j + 1) % segments + 1
            tris.append((a, b, d))
            tris.append((a, d, c))
    return verts, tris


def _build_octahedron(size=1.0):
    s = size
    verts = [
        ( 0,  s,  0),
        ( s,  0,  0), ( 0,  0,  s),
        (-s,  0,  0), ( 0,  0, -s),
        ( 0, -s,  0),
    ]
    tris = [
        (1,2,3), (1,3,4), (1,4,5), (1,5,2),
        (6,3,2), (6,4,3), (6,5,4), (6,2,5),
    ]
    return verts, tris


_SHAPE_BUILDERS = {
    "cube":       _build_cube,
    "pyramid":    _build_pyramid,
    "sphere":     _build_sphere,
    "cylinder":   _build_cylinder,
    "torus":      _build_torus,
    "octahedron": _build_octahedron,
}


def export_obj(shape_type, size=1.0, filename="export.obj"):
    """Export the given shape to a Wavefront OBJ file.

    Generates accurate vertex and triangle face data for all 6 supported
    shapes.  Returns True on success, False on error.
    """
    builder = _SHAPE_BUILDERS.get(shape_type)
    if builder is None:
        print(f">> ⚠️  Unknown shape '{shape_type}' for OBJ export. Using cube.")
        builder = _build_cube

    try:
        verts, tris = builder(size)
        with open(filename, "w") as f:
            f.write(f"# PROJECT VERTEX Export\n")
            f.write(f"# Shape : {shape_type}\n")
            f.write(f"# Date  : {datetime.now().isoformat(timespec='seconds')}\n\n")
            f.write(f"o {shape_type}\n\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")
            for tri in tris:
                f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")
        print(f">> 📤 EXPORTED: {filename}  ({len(verts)} verts, {len(tris)} faces)")
        return True
    except Exception as e:
        print(f">> ❌ EXPORT ERROR: {e}")
        return False


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def clamp(value, min_val, max_val):
    """Clamp *value* to [min_val, max_val]."""
    return max(min_val, min(value, max_val))


def lerp(start, end, factor):
    """Linear interpolation between *start* and *end* by *factor* ∈ [0, 1]."""
    return start + (end - start) * factor
