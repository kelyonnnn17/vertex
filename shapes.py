"""
3D Shape definitions and rendering functions — Project Vertex
Shapes: cube, pyramid, sphere, cylinder, torus, octahedron,
        cone, diamond, icosahedron, torus_knot
"""
from OpenGL.GL import *
import math


class ShapeRenderer:
    def __init__(self, wireframe=False):
        self.wireframe = wireframe

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

    @staticmethod
    def _normalize(v):
        l = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        return (v[0]/l, v[1]/l, v[2]/l) if l else (0.0, 1.0, 0.0)

    @classmethod
    def _face_normal(cls, verts, indices):
        v0, v1, v2 = verts[indices[0]], verts[indices[1]], verts[indices[2]]
        a = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
        b = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
        return cls._normalize(cls._cross(a, b))

    # ------------------------------------------------------------------
    # Original shapes (improved normals + lighting)
    # ------------------------------------------------------------------

    def draw_cube(self, size=1.0):
        s = size
        verts = [
            ( s,-s,-s),( s, s,-s),(-s, s,-s),(-s,-s,-s),
            ( s,-s, s),( s, s, s),(-s,-s, s),(-s, s, s),
        ]
        faces_normals = [
            ((0,1,2,3), ( 0, 0,-1)), ((4,7,6,5), ( 0, 0, 1)),
            ((0,4,5,1), ( 1, 0, 0)), ((2,6,7,3), (-1, 0, 0)),
            ((0,3,7,4), ( 0,-1, 0)), ((1,5,6,2), ( 0, 1, 0)),
        ]
        edges = [(0,1),(0,3),(0,4),(2,1),(2,3),(2,7),(6,3),(6,4),(6,7),(5,1),(5,4),(5,7)]
        if self.wireframe:
            glBegin(GL_LINES)
            for e in edges:
                for v in e: glVertex3fv(verts[v])
            glEnd()
        else:
            glBegin(GL_QUADS)
            for idxs, n in faces_normals:
                glNormal3fv(n)
                for v in idxs: glVertex3fv(verts[v])
            glEnd()

    def draw_pyramid(self, size=1.0):
        s = size
        verts = [(0,s,0),(-s,-s,s),(s,-s,s),(s,-s,-s),(-s,-s,-s)]
        sides = [(0,1,2),(0,2,3),(0,3,4),(0,4,1)]
        edges = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
        if self.wireframe:
            glBegin(GL_LINES)
            for e in edges:
                for v in e: glVertex3fv(verts[v])
            glEnd()
        else:
            glBegin(GL_TRIANGLES)
            for f in sides:
                glNormal3fv(self._face_normal(verts, f))
                for v in f: glVertex3fv(verts[v])
            glEnd()
            glBegin(GL_QUADS)
            glNormal3f(0,-1,0)
            for v in (1,2,3,4): glVertex3fv(verts[v])
            glEnd()

    def draw_sphere(self, radius=1.0, slices=24, stacks=24):
        if self.wireframe:
            glBegin(GL_LINES)
            for i in range(slices):
                a1 = 2*math.pi*i/slices
                a2 = 2*math.pi*(i+1)/slices
                for j in range(stacks):
                    p1 = math.pi*j/stacks
                    p2 = math.pi*(j+1)/stacks
                    x1=radius*math.sin(p1)*math.cos(a1); y1=radius*math.cos(p1); z1=radius*math.sin(p1)*math.sin(a1)
                    x1b=radius*math.sin(p2)*math.cos(a1);y1b=radius*math.cos(p2);z1b=radius*math.sin(p2)*math.sin(a1)
                    x2=radius*math.sin(p1)*math.cos(a2); y2=radius*math.cos(p1); z2=radius*math.sin(p1)*math.sin(a2)
                    glVertex3f(x1,y1,z1); glVertex3f(x1b,y1b,z1b)
                    glVertex3f(x1,y1,z1); glVertex3f(x2,y2,z2)
            glEnd()
        else:
            for i in range(stacks):
                glBegin(GL_QUAD_STRIP)
                for j in range(slices+1):
                    for k in range(2):
                        phi=math.pi*(i+k)/stacks; theta=2*math.pi*j/slices
                        x=math.sin(phi)*math.cos(theta); y=math.cos(phi); z=math.sin(phi)*math.sin(theta)
                        glNormal3f(x,y,z); glVertex3f(radius*x,radius*y,radius*z)
                glEnd()

    def draw_cylinder(self, radius=1.0, height=2.0, segments=24):
        h = height/2
        if self.wireframe:
            glBegin(GL_LINES)
            for i in range(segments):
                a=2*math.pi*i/segments; x=radius*math.cos(a); z=radius*math.sin(a)
                glVertex3f(x,-h,z); glVertex3f(x,h,z)
            for y in [-h,h]:
                for i in range(segments):
                    a1=2*math.pi*i/segments; a2=2*math.pi*(i+1)/segments
                    glVertex3f(radius*math.cos(a1),y,radius*math.sin(a1))
                    glVertex3f(radius*math.cos(a2),y,radius*math.sin(a2))
            glEnd()
        else:
            for y_cap, ny, winding in [(h, 1, 1), (-h, -1, -1)]:
                glBegin(GL_TRIANGLE_FAN)
                glNormal3f(0,ny,0); glVertex3f(0,y_cap,0)
                rng = range(segments+1) if ny>0 else range(segments,-1,-1)
                for i in rng:
                    a=2*math.pi*i/segments; glVertex3f(radius*math.cos(a),y_cap,radius*math.sin(a))
                glEnd()
            glBegin(GL_QUAD_STRIP)
            for i in range(segments+1):
                a=2*math.pi*i/segments; nx=math.cos(a); nz=math.sin(a)
                glNormal3f(nx,0,nz)
                glVertex3f(radius*nx,h,radius*nz); glVertex3f(radius*nx,-h,radius*nz)
            glEnd()

    def draw_torus(self, inner_radius=0.3, outer_radius=1.0, segments=28, rings=28):
        if self.wireframe:
            glBegin(GL_LINES)
            for i in range(rings):
                for j in range(segments):
                    def _tp(u,v):
                        r=outer_radius+inner_radius*math.cos(v)
                        return (r*math.cos(u),inner_radius*math.sin(v),r*math.sin(u))
                    u0=2*math.pi*i/rings; u1=2*math.pi*(i+1)/rings
                    v0=2*math.pi*j/segments; v1=2*math.pi*(j+1)/segments
                    glVertex3fv(_tp(u0,v0)); glVertex3fv(_tp(u1,v0))
                    glVertex3fv(_tp(u0,v0)); glVertex3fv(_tp(u0,v1))
            glEnd()
        else:
            for i in range(rings):
                glBegin(GL_QUAD_STRIP)
                for j in range(segments+1):
                    for k in range(2):
                        u=2*math.pi*(i+k)/rings; v=2*math.pi*j/segments
                        cv=math.cos(v); sv=math.sin(v)
                        r=outer_radius+inner_radius*cv
                        glNormal3f(cv*math.cos(u),sv,cv*math.sin(u))
                        glVertex3f(r*math.cos(u),inner_radius*sv,r*math.sin(u))
                glEnd()

    def draw_octahedron(self, size=1.0):
        s = size
        verts = [(0,s,0),(s,0,0),(0,0,s),(-s,0,0),(0,0,-s),(0,-s,0)]
        # Flipped face winding so cross-product normals point OUTWARD
        faces = [(0,2,1),(0,3,2),(0,4,3),(0,1,4),(5,1,2),(5,2,3),(5,3,4),(5,4,1)]
        edges = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1),(5,1),(5,2),(5,3),(5,4)]
        if self.wireframe:
            glBegin(GL_LINES)
            for e in edges:
                for v in e: glVertex3fv(verts[v])
            glEnd()
        else:
            glBegin(GL_TRIANGLES)
            for f in faces:
                glNormal3fv(self._face_normal(verts, f))
                for v in f: glVertex3fv(verts[v])
            glEnd()

    # ------------------------------------------------------------------
    # New shapes
    # ------------------------------------------------------------------

    def draw_cone(self, radius=1.0, height=2.0, segments=28):
        """Draw a cone: flat base + pointed tip."""
        h = height / 2
        slant = math.sqrt(radius**2 + height**2)  # for proper normals
        ny_side = radius / slant   # outward Y component of side normal
        nr_side = height / slant   # outward R component

        if self.wireframe:
            # Tip to base edges + base ring
            tip = (0, h, 0)
            glBegin(GL_LINES)
            for i in range(segments):
                a1 = 2*math.pi*i/segments
                a2 = 2*math.pi*(i+1)/segments
                base1 = (radius*math.cos(a1), -h, radius*math.sin(a1))
                base2 = (radius*math.cos(a2), -h, radius*math.sin(a2))
                glVertex3fv(tip); glVertex3fv(base1)
                glVertex3fv(base1); glVertex3fv(base2)
            glEnd()
        else:
            # Side surface
            glBegin(GL_TRIANGLES)
            for i in range(segments):
                a1 = 2*math.pi*i/segments
                a2 = 2*math.pi*(i+1)/segments
                # Tip normal: average of adjacent side normals
                na = math.cos((a1+a2)/2)*nr_side
                nb = ny_side
                nc = math.sin((a1+a2)/2)*nr_side
                glNormal3f(na, nb, nc)
                glVertex3f(0, h, 0)
                glNormal3f(math.cos(a1)*nr_side, ny_side, math.sin(a1)*nr_side)
                glVertex3f(radius*math.cos(a1), -h, radius*math.sin(a1))
                glNormal3f(math.cos(a2)*nr_side, ny_side, math.sin(a2)*nr_side)
                glVertex3f(radius*math.cos(a2), -h, radius*math.sin(a2))
            glEnd()
            # Base cap
            glBegin(GL_TRIANGLE_FAN)
            glNormal3f(0, -1, 0); glVertex3f(0, -h, 0)
            for i in range(segments, -1, -1):
                a = 2*math.pi*i/segments
                glVertex3f(radius*math.cos(a), -h, radius*math.sin(a))
            glEnd()

    def draw_diamond(self, size=1.0):
        """Elongated bipyramid — gem-like diamond shape."""
        w = size * 0.6   # equatorial radius
        tip_t = size * 1.3  # top tip height
        tip_b = size * 0.8  # bottom tip height
        segs = 8            # octagonal cross-section

        if self.wireframe:
            glBegin(GL_LINES)
            for i in range(segs):
                a1 = 2*math.pi*i/segs
                a2 = 2*math.pi*(i+1)/segs
                x1, z1 = w*math.cos(a1), w*math.sin(a1)
                x2, z2 = w*math.cos(a2), w*math.sin(a2)
                # equatorial ring
                glVertex3f(x1, 0, z1); glVertex3f(x2, 0, z2)
                # top lines
                glVertex3f(0, tip_t, 0); glVertex3f(x1, 0, z1)
                # bottom lines
                glVertex3f(0, -tip_b, 0); glVertex3f(x1, 0, z1)
            glEnd()
        else:
            glBegin(GL_TRIANGLES)
            for i in range(segs):
                a1 = 2*math.pi*i/segs
                a2 = 2*math.pi*(i+1)/segs
                x1, z1 = w*math.cos(a1), w*math.sin(a1)
                x2, z2 = w*math.cos(a2), w*math.sin(a2)
                # top face (wind: top -> a2 -> a1 for outward normal)
                top = (0, tip_t, 0)
                n = self._face_normal([top, (x2,0,z2), (x1,0,z1)], [0,1,2])
                glNormal3fv(n)
                glVertex3fv(top); glVertex3f(x2,0,z2); glVertex3f(x1,0,z1)
                # bottom face (wind: bot -> a1 -> a2 for outward normal)
                bot = (0, -tip_b, 0)
                n = self._face_normal([bot, (x1,0,z1), (x2,0,z2)], [0,1,2])
                glNormal3fv(n)
                glVertex3fv(bot); glVertex3f(x1,0,z1); glVertex3f(x2,0,z2)
            glEnd()

    def draw_icosahedron(self, size=1.0):
        """Regular icosahedron — 12 vertices, 20 equilateral triangle faces."""
        phi = (1 + math.sqrt(5)) / 2  # golden ratio ≈ 1.618
        # Vertices (unnormalized, then scaled)
        raw = [
            (-1, phi,0),(1, phi,0),(-1,-phi,0),(1,-phi,0),
            (0,-1, phi),(0, 1, phi),(0,-1,-phi),(0, 1,-phi),
            ( phi,0,-1),(phi,0, 1),(-phi,0,-1),(-phi,0, 1),
        ]
        scale = size / math.sqrt(1 + phi**2)
        verts = [(x*scale, y*scale, z*scale) for x,y,z in raw]

        faces = [
            (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
            (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
            (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
            (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
        ]
        edges_set = set()
        for f in faces:
            for i in range(3):
                e = tuple(sorted((f[i], f[(i+1)%3])))
                edges_set.add(e)

        if self.wireframe:
            glBegin(GL_LINES)
            for a, b in edges_set:
                glVertex3fv(verts[a]); glVertex3fv(verts[b])
            glEnd()
        else:
            glBegin(GL_TRIANGLES)
            for f in faces:
                n = self._face_normal(verts, f)
                glNormal3fv(n)
                for v in f: glVertex3fv(verts[v])
            glEnd()

    def draw_torus_knot(self, size=1.0, p=2, q=3, tube_r=0.18, curve_pts=180, tube_segs=18):
        """Trefoil torus knot (default p=2, q=3) swept as a tube.

        The centerline follows: (cos(pt)*(R+cos(qt)), sin(qt), sin(pt)*(R+cos(qt)))
        with a rotation-minimizing Frenet frame for the tube cross-section.
        """
        R = size * 0.65        # major radius
        tube_r = tube_r * size

        def knot_pt(t):
            return (
                math.cos(p*t) * (R + math.cos(q*t)),
                math.sin(q*t) * size * 0.5,
                math.sin(p*t) * (R + math.cos(q*t)),
            )

        # Build centerline + tangents
        pts = []
        tans = []
        dt = 2*math.pi / curve_pts
        for i in range(curve_pts):
            t = 2*math.pi * i / curve_pts
            pts.append(knot_pt(t))
            # Forward difference for tangent
            p_next = knot_pt(t + dt)
            tx = p_next[0]-pts[-1][0]; ty = p_next[1]-pts[-1][1]; tz = p_next[2]-pts[-1][2]
            tans.append(self._normalize((tx, ty, tz)))

        # Build rotation-minimizing frames
        # Seed: first normal perpendicular to first tangent
        T0 = tans[0]
        # Pick an arbitrary vector not parallel to T0
        arb = (0, 1, 0) if abs(T0[1]) < 0.9 else (1, 0, 0)
        N0 = self._normalize(self._cross(T0, arb))
        frames = [(T0, N0, self._normalize(self._cross(T0, N0)))]

        for i in range(1, curve_pts):
            T_prev, N_prev, B_prev = frames[-1]
            T_cur = tans[i]
            # Rotate N_prev by the rotation that maps T_prev→T_cur
            axis = self._cross(T_prev, T_cur)
            axis_len = math.sqrt(sum(x*x for x in axis))
            if axis_len > 1e-8:
                axis = (axis[0]/axis_len, axis[1]/axis_len, axis[2]/axis_len)
                angle = math.acos(max(-1, min(1, sum(a*b for a,b in zip(T_prev, T_cur)))))
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                # Rodrigues rotation of N_prev around axis
                dot = sum(axis[k]*N_prev[k] for k in range(3))
                cross = self._cross(axis, N_prev)
                N_cur = self._normalize((
                    N_prev[0]*cos_a + cross[0]*sin_a + axis[0]*dot*(1-cos_a),
                    N_prev[1]*cos_a + cross[1]*sin_a + axis[1]*dot*(1-cos_a),
                    N_prev[2]*cos_a + cross[2]*sin_a + axis[2]*dot*(1-cos_a),
                ))
            else:
                N_cur = N_prev
            B_cur = self._normalize(self._cross(T_cur, N_cur))
            frames.append((T_cur, N_cur, B_cur))

        # Generate tube rings
        rings = []
        for i in range(curve_pts):
            cx, cy, cz = pts[i]
            _, N, B = frames[i]
            ring = []
            for j in range(tube_segs):
                theta = 2*math.pi * j / tube_segs
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                nx = N[0]*cos_t + B[0]*sin_t
                ny = N[1]*cos_t + B[1]*sin_t
                nz = N[2]*cos_t + B[2]*sin_t
                ring.append((cx + tube_r*nx, cy + tube_r*ny, cz + tube_r*nz, nx, ny, nz))
            rings.append(ring)

        if self.wireframe:
            glBegin(GL_LINES)
            for i in range(curve_pts):
                ni = (i+1) % curve_pts
                for j in range(tube_segs):
                    nj = (j+1) % tube_segs
                    v0 = rings[i][j]; v1 = rings[ni][j]; v2 = rings[i][nj]
                    glVertex3f(v0[0],v0[1],v0[2]); glVertex3f(v1[0],v1[1],v1[2])
                    glVertex3f(v0[0],v0[1],v0[2]); glVertex3f(v2[0],v2[1],v2[2])
            glEnd()
        else:
            for i in range(curve_pts):
                ni = (i+1) % curve_pts
                glBegin(GL_QUAD_STRIP)
                for j in range(tube_segs+1):
                    jj = j % tube_segs
                    v0 = rings[i][jj]; v1 = rings[ni][jj]
                    glNormal3f(v0[3],v0[4],v0[5]); glVertex3f(v0[0],v0[1],v0[2])
                    glNormal3f(v1[3],v1[4],v1[5]); glVertex3f(v1[0],v1[1],v1[2])
                glEnd()

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    SHAPE_NAMES = [
        "cube", "pyramid", "sphere", "cylinder", "torus", "octahedron",
        "cone", "diamond", "icosahedron", "torus_knot",
    ]

    def draw_shape(self, shape_type, size=1.0):
        """Draw a shape by name. Falls back to cube for unknown names."""
        dispatch = {
            "cube":        lambda: self.draw_cube(size),
            "pyramid":     lambda: self.draw_pyramid(size),
            "sphere":      lambda: self.draw_sphere(size),
            "cylinder":    lambda: self.draw_cylinder(size, size*2),
            "torus":       lambda: self.draw_torus(size*0.3, size),
            "octahedron":  lambda: self.draw_octahedron(size),
            "cone":        lambda: self.draw_cone(size, size*2),
            "diamond":     lambda: self.draw_diamond(size),
            "icosahedron": lambda: self.draw_icosahedron(size),
            "torus_knot":  lambda: self.draw_torus_knot(size),
        }
        dispatch.get(shape_type, lambda: self.draw_cube(size))()

    def get_shape_info(self, shape_type):
        """Return approximate (vertices, faces) counts for status bar."""
        info = {
            "cube":       (8,  6),  "pyramid":    (5,  5),
            "sphere":     (576, 552),"cylinder":  (50, 50),
            "torus":      (784, 756),"octahedron": (6, 8),
            "cone":       (29, 28), "diamond":    (10, 16),
            "icosahedron":(12, 20), "torus_knot": (3240, 3060),
        }
        return info.get(shape_type, (0, 0))
