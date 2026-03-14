"""
OpenGL Renderer — single unified window for Project Vertex.
Renders the 3D scene, a webcam PiP overlay, gesture guide panel, and status bar.
"""
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
import time
import ctypes
import numpy as np

from shapes import ShapeRenderer
from gesture_engine import GestureMode, MODE_META, GESTURE_GUIDE


# ---------------------------------------------------------------------------
# Colour palette (12 distinct, name + RGB)
# ---------------------------------------------------------------------------
COLOR_PALETTE = [
    ("CYAN",    (0.00, 1.00, 1.00)),
    ("SKY",     (0.00, 0.55, 1.00)),
    ("PURPLE",  (0.60, 0.00, 1.00)),
    ("MAGENTA", (1.00, 0.00, 0.80)),
    ("RED",     (1.00, 0.15, 0.10)),
    ("ORANGE",  (1.00, 0.50, 0.00)),
    ("YELLOW",  (1.00, 0.95, 0.00)),
    ("LIME",    (0.45, 1.00, 0.00)),
    ("GREEN",   (0.00, 1.00, 0.30)),
    ("TEAL",    (0.00, 0.85, 0.60)),
    ("WHITE",   (0.90, 0.90, 0.90)),
    ("GOLD",    (1.00, 0.80, 0.20)),
]

# Background themes (R,G,B,A)
BG_THEMES = [
    (0.04, 0.04, 0.10, 1.0),  # Dark blue (default)
    (0.00, 0.00, 0.00, 1.0),  # Black
    (0.04, 0.10, 0.04, 1.0),  # Dark green
    (0.10, 0.03, 0.12, 1.0),  # Deep purple
    (0.08, 0.05, 0.02, 1.0),  # Dark amber
]

# Shared font and texture caches
_FONTS: dict[str, pygame.font.Font] = {}
_TEXT_TEX_CACHE: dict[tuple, tuple] = {}


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    key = f"{size}_{bold}"
    if key not in _FONTS:
        pygame.font.init()
        # Use a cleaner, modern sans-serif font instead of Courier
        font_name = pygame.font.match_font('avenirnext,trebuchetms,helvetica,arial')
        _FONTS[key] = pygame.font.Font(font_name, size)
        if bold:
            _FONTS[key].set_bold(True)
    return _FONTS[key]


class Renderer:
    def __init__(self, config):
        self.config = config
        self.width  = config.get("display", "width")
        self.height = config.get("display", "height")

        self.shape_renderer = ShapeRenderer(wireframe=config.get("rendering", "wireframe"))

        self.fps         = 0
        self.frame_count = 0
        self.last_fps_t  = time.time()

        # Webcam PiP texture
        self._pip_tex: int = 0          # OpenGL texture id (0 = not yet created)
        self._pip_w   = 0
        self._pip_h   = 0

        # Show/hide gesture guide (toggled with ? key)
        self.show_guide = True

        self.init_opengl()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_opengl(self):
        glEnable(GL_DEPTH_TEST);  glDepthFunc(GL_LEQUAL)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Key light (upper-right, warm)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [2, 3, 2, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.20, 0.20, 0.22, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.85, 0.82, 0.78, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.00, 0.95, 0.90, 1])

        # Fill light (lower-left, cool)
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, [-2, -1, -1, 0])
        glLightfv(GL_LIGHT1, GL_AMBIENT,  [0.00, 0.00, 0.00, 1])
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  [0.20, 0.22, 0.30, 1])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.00, 0.00, 0.00, 1])

        # Material shininess
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 48)

        self._apply_bg()
        self.setup_camera()

    def setup_camera(self):
        fov  = self.config.get("display", "fov")
        near = self.config.get("display", "near")
        far  = self.config.get("display", "far")
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(fov, self.width / self.height, near, far)
        glMatrixMode(GL_MODELVIEW)

    # ------------------------------------------------------------------
    # 3-D scene elements
    # ------------------------------------------------------------------

    def draw_grid(self):
        if not self.config.get("rendering", "show_grid"):
            return
        c = self.config.get("rendering", "grid_color")
        glColor4f(c[0], c[1], c[2], 0.4)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBegin(GL_LINES)
        size, step = 10, 1.0
        for i in range(-size, size+1):
            glVertex3f(i*step, 0, -size*step); glVertex3f(i*step, 0,  size*step)
            glVertex3f(-size*step, 0, i*step); glVertex3f( size*step, 0, i*step)
        glEnd()
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def draw_axes(self):
        if not self.config.get("rendering", "show_axes"):
            return
        glDisable(GL_LIGHTING); glLineWidth(2.5)
        glBegin(GL_LINES)
        for axis_col, end in [((1,0,0),(2,0,0)),((0,1,0),(0,2,0)),((0,0,1),(0,0,2))]:
            glColor3fv(axis_col); glVertex3f(0,0,0); glVertex3fv(end)
        glEnd()
        glLineWidth(1.0); glEnable(GL_LIGHTING)

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def render_frame(self, screen, shape_type, rot_x, rot_y, zoom_level,
                     shape_scale=1.0, color_idx=0,
                     current_mode=GestureMode.NONE,
                     webcam_frame=None, pip_visible=True,
                     auto_rotate=False):
        """Render one full frame — 3D scene + all 2D overlays."""

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

        # 3-D scene
        glTranslatef(0.0, 0.0, zoom_level)
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)

        self.draw_grid()
        self.draw_axes()

        color = COLOR_PALETTE[color_idx][1]
        glColor3f(*color)
        glPushMatrix()
        glScalef(shape_scale, shape_scale, shape_scale)
        self.shape_renderer.draw_shape(shape_type)
        glPopMatrix()

        # ---- 2-D overlay pass ----
        self._enter_2d()

        # Webcam PiP (bottom-right)
        if pip_visible and webcam_frame is not None:
            self._render_pip(webcam_frame)

        # Gesture guide panel (left side)
        if self.show_guide:
            self._draw_gesture_guide(current_mode)

        # Status bar (bottom strip)
        v, f = self.shape_renderer.get_shape_info(shape_type)
        self._draw_status_bar(shape_type, color_idx, rot_x, rot_y,
                              zoom_level, shape_scale, auto_rotate, v, f)

        # HUD (top-left)
        if self.config.get("rendering", "show_hud"):
            self._draw_hud(shape_type, rot_x, rot_y, zoom_level, color_idx, current_mode)

        self._exit_2d()

        # FPS
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_t >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_t  = now

    # ------------------------------------------------------------------
    # PiP webcam
    # ------------------------------------------------------------------

    def _render_pip(self, bgr_frame):
        import cv2
        h, w = bgr_frame.shape[:2]

        # Scale to fit pip box
        pip_w = int(self.width * 0.27)
        pip_h = int(pip_w * h / w)
        x = self.width  - pip_w - 12
        y = self.height - pip_h - 48   # leave room for status bar

        # Upload texture (convert BGR→RGB, flip Y for OpenGL)
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (pip_w, pip_h))

        if self._pip_tex == 0:
            self._pip_tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self._pip_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, pip_w, pip_h, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, rgb.tobytes())

        # Draw border
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glColor4f(0, 0.85, 0.85, 0.8)
        glLineWidth(2)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x-2,   y-2)
        glVertex2f(x+pip_w+2, y-2)
        glVertex2f(x+pip_w+2, y+pip_h+2)
        glVertex2f(x-2,   y+pip_h+2)
        glEnd()
        glLineWidth(1)

        # Draw textured quad
        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0,0); glVertex2f(x,      y)
        glTexCoord2f(1,0); glVertex2f(x+pip_w, y)
        glTexCoord2f(1,1); glVertex2f(x+pip_w, y+pip_h)
        glTexCoord2f(0,1); glVertex2f(x,      y+pip_h)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)

        # "CAM" label
        self._blit_text("CAM", x+5, y+5, 13, (0, 220, 220), bold=True)

    # ------------------------------------------------------------------
    # Gesture guide panel
    # ------------------------------------------------------------------

    def _draw_gesture_guide(self, current_mode):
        panel_x, panel_y = 10, 10
        line_h = 22
        panel_w = 235
        panel_h = len(GESTURE_GUIDE) * line_h + 30

        # Semi-transparent background
        glEnable(GL_BLEND)
        glColor4f(0, 0, 0, 0.55)
        glBegin(GL_QUADS)
        glVertex2f(panel_x,          panel_y)
        glVertex2f(panel_x+panel_w,  panel_y)
        glVertex2f(panel_x+panel_w,  panel_y+panel_h)
        glVertex2f(panel_x,          panel_y+panel_h)
        glEnd()
        glDisable(GL_BLEND)

        # Title
        self._blit_text("GESTURE GUIDE [?]", panel_x+8, panel_y+6, 13,
                        (0, 220, 220), bold=True)

        for i, (gesture_str, action_str) in enumerate(GESTURE_GUIDE):
            yy = panel_y + 24 + i * line_h
            # Highlight active mode row
            mode_for_row = [
                GestureMode.NONE, GestureMode.ROTATE, GestureMode.PAN,
                GestureMode.COLOR, GestureMode.SCALE, GestureMode.SHAPE,
                GestureMode.ZOOM,
            ][i] if i < 7 else None

            if mode_for_row == current_mode:
                glEnable(GL_BLEND)
                glColor4f(0, 0.6, 0.6, 0.3)
                glBegin(GL_QUADS)
                glVertex2f(panel_x+2, yy-2)
                glVertex2f(panel_x+panel_w-2, yy-2)
                glVertex2f(panel_x+panel_w-2, yy+line_h-2)
                glVertex2f(panel_x+2, yy+line_h-2)
                glEnd()
                glDisable(GL_BLEND)
                col = (0, 255, 200)
            else:
                col = (160, 160, 160)

            self._blit_text(f"{gesture_str}  →  {action_str}",
                            panel_x+8, yy, 12, col)

    # ------------------------------------------------------------------
    # HUD (top-left, top of guide panel)
    # ------------------------------------------------------------------

    def _draw_hud(self, shape_type, rot_x, rot_y, zoom, color_idx,
                  current_mode):
        mode_label, mode_rgb = MODE_META[current_mode]
        col_name, col_rgb    = COLOR_PALETTE[color_idx]

        # Moved further right to avoid overlapping the gesture guide
        panel_x = 280
        lines = [
            (f"FPS:    {self.fps}",                       (120, 220, 120)),
            (f"SHAPE:  {shape_type.upper()}",             (200, 200, 200)),
            (f"COLOR:  {col_name}",                       tuple(int(v*255) for v in col_rgb)),
            (f"ROT:    X{rot_x:+.0f}°  Y{rot_y:+.0f}°",  (180, 180, 220)),
            (f"ZOOM:   {zoom:.2f}",                       (180, 220, 180)),
            (f"MODE:   {mode_label}",                     mode_rgb),
        ]
        for i, (txt, col) in enumerate(lines):
            self._blit_text(txt, panel_x, 16 + i*22, 16, col, bold=(i==0))

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _draw_status_bar(self, shape_type, color_idx, rot_x, rot_y,
                         zoom, scale, auto_rotate, verts, faces):
        bar_h = 34
        y0    = self.height - bar_h

        glEnable(GL_BLEND)
        glColor4f(0, 0, 0, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(0,           y0)
        glVertex2f(self.width,  y0)
        glVertex2f(self.width,  self.height)
        glVertex2f(0,           self.height)
        glEnd()
        glDisable(GL_BLEND)

        # Divider line
        glColor4f(0, 0.7, 0.7, 0.6)
        glBegin(GL_LINES)
        glVertex2f(0, y0); glVertex2f(self.width, y0)
        glEnd()

        col_name = COLOR_PALETTE[color_idx][0]
        ar_str   = "AUTO-ROT" if auto_rotate else ""
        segments = [
            f"  {shape_type.upper()}",
            f"  COLOR:{col_name}",
            f"  SCALE:{scale:.2f}",
            f"  ROT X{rot_x:+.0f}° Y{rot_y:+.0f}°",
            f"  ZOOM:{zoom:.1f}",
            f"  V:{verts}  F:{faces}",
            f"  {ar_str}",
        ]
        sep_col = (0, 100, 100)
        txt_x   = 5
        ty      = y0 + 10
        for seg in segments:
            self._blit_text(seg, txt_x, ty, 13, (180, 220, 220))
            txt_x += len(seg) * 8 + 4

        # Key hints on the right
        hints = "[1-0] Shape  [C] Color  [R] Auto-rot  [B] BG  [W] Wire  [P] Screenshot  [?] Guide  [ESC] Quit"
        self._blit_text(hints, 5, y0 + 20, 11, (100, 120, 120))

    # ------------------------------------------------------------------
    # 2-D pass helpers
    # ------------------------------------------------------------------

    def _enter_2d(self):
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)

    def _exit_2d(self):
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW); glPopMatrix()

    def _blit_text(self, text, x, y, size, color_rgb, bold=False):
        """Render text at (x,y) in screen-space using an OpenGL texture cache."""
        key = (text, size, color_rgb, bold)
        if key not in _TEXT_TEX_CACHE:
            surf = _font(size, bold).render(text, True, color_rgb)
            tw, th = surf.get_size()
            try:
                data = pygame.image.tobytes(surf, "RGBA", True)
            except AttributeError:
                data = pygame.image.tostring(surf, "RGBA", True)
            
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
            _TEXT_TEX_CACHE[key] = (tex, tw, th)

        tex, tw, th = _TEXT_TEX_CACHE[key]
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, tex)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        # Flip the V coordinate to render the text right-side up
        glTexCoord2f(0, 1); glVertex2f(x, y)
        glTexCoord2f(1, 1); glVertex2f(x+tw, y)
        glTexCoord2f(1, 0); glVertex2f(x+tw, y+th)
        glTexCoord2f(0, 0); glVertex2f(x, y+th)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def set_background(self, theme_idx):
        self.config.set("rendering", "background_color",
                        value=list(BG_THEMES[theme_idx % len(BG_THEMES)]))
        self._apply_bg()

    def _apply_bg(self):
        bg = self.config.get("rendering", "background_color")
        glClearColor(bg[0], bg[1], bg[2], bg[3])
