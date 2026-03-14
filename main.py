"""
PROJECT VERTEX — Main Application
Single-window 3D workbench with hand-gesture control.
"""
import pygame
from pygame.locals import *
from OpenGL.GL import *
import cv2
import math
import os

from vision import HandSensor
from config import Config
from renderer import Renderer, COLOR_PALETTE, BG_THEMES
from utils import save_blueprint, load_blueprint, clamp, lerp, export_obj, screenshot
from shapes import ShapeRenderer
from gesture_engine import GestureMode


class VertexApp:
    def __init__(self):
        self.config = Config()

        # ---- pygame + OpenGL window ----
        pygame.init()
        self.W = self.config.get("display", "width")
        self.H = self.config.get("display", "height")
        self.screen = pygame.display.set_mode(
            (self.W, self.H), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("PROJECT VERTEX  |  WORKBENCH  v3.0")

        self.renderer = Renderer(self.config)
        self.renderer.setup_camera()

        # ---- webcam ----
        self.cap = None
        self._init_webcam()

        # ---- hand sensor ----
        dc = self.config.get("hand_sensor", "detection_confidence")
        tc = self.config.get("hand_sensor", "tracking_confidence")
        self.sensor = HandSensor(dc, tc)

        # ---- view state ----
        self.rot_x        = 0.0
        self.rot_y        = 0.0
        self.target_rot_x = 0.0
        self.target_rot_y = 0.0
        self.zoom_level   = self.config.get("camera", "initial_z")
        self.shape_scale  = self.config.get("rendering", "shape_scale") or 1.0

        # ---- shape / color ----
        self.shapes       = ShapeRenderer.SHAPE_NAMES
        self.shape_idx    = self._shape_idx(self.config.get("shapes", "default"))
        self.color_idx    = self.config.get("rendering", "color_index") or 0
        self.bg_theme_idx = 0

        # ---- render flags ----
        self.wireframe   = self.config.get("rendering", "wireframe")
        self.show_grid   = self.config.get("rendering", "show_grid")
        self.show_axes   = self.config.get("rendering", "show_axes")
        self.show_hud    = self.config.get("rendering", "show_hud")
        self.pip_visible = True
        self.auto_rotate = False
        self.auto_rot_speed = 0.4

        # ---- gesture state ----
        self._pinch_last_pos  = None
        self._zoom_base_dist  = None
        self._zoom_base_level = self.zoom_level
        self._current_mode    = GestureMode.NONE
        self._pan_last        = None

        self.running = True
        self._print_banner()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_webcam(self):
        for cam_idx in range(3):
            cap = cv2.VideoCapture(cam_idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap = cap
                print(f">> Webcam opened (index {cam_idx})")
                return
        print(">> WARNING: No webcam found — hand tracking disabled.")

    def _shape_idx(self, name: str) -> int:
        try:    return self.shapes.index(name)
        except: return 0

    def _print_banner(self):
        print(">> ENGINE ONLINE  —  PROJECT VERTEX v3.0")
        print(">> ─────────────────────────────────────────")
        print(">> KEYBOARD:")
        print("   [1-0] Shape  [W] Wireframe  [G] Grid  [A] Axes")
        print("   [H] HUD  [C] Color  [R] Auto-rotate  [B] Background")
        print("   [+/-] Scale  [P] Screenshot  [S] Save  [L] Load")
        print("   [E] Export OBJ  [?] Gesture guide  [ESC] Quit")
        print(">> GESTURES:")
        print("   Fist hold -> Reset   1 finger+pinch -> Orbit")
        print("   2 fingers -> Pan     3 fingers wave -> Color")
        print("   4 fingers -> Scale   5 fingers wave -> Shape")
        print("   2 hands open -> Zoom")

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)

    def _handle_key(self, key):
        shape_keys = {
            pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3,
            pygame.K_5: 4, pygame.K_6: 5, pygame.K_7: 6, pygame.K_8: 7,
            pygame.K_9: 8, pygame.K_0: 9,
        }
        if key in shape_keys:
            idx = shape_keys[key]
            if idx < len(self.shapes):
                self.shape_idx = idx
            return

        actions = {
            pygame.K_w:       self._toggle_wireframe,
            pygame.K_g:       self._toggle_grid,
            pygame.K_a:       self._toggle_axes,
            pygame.K_h:       self._toggle_hud,
            pygame.K_c:       self._cycle_color,
            pygame.K_r:       self._toggle_auto_rotate,
            pygame.K_b:       self._cycle_bg,
            pygame.K_PLUS:    lambda: self._adjust_scale(+0.1),
            pygame.K_EQUALS:  lambda: self._adjust_scale(+0.1),
            pygame.K_MINUS:   lambda: self._adjust_scale(-0.1),
            pygame.K_p:       self._take_screenshot,
            pygame.K_s:       self._save,
            pygame.K_l:       self._load,
            pygame.K_e:       self._export,
            pygame.K_SLASH:   self._toggle_guide,
            pygame.K_QUESTION:self._toggle_guide,
            pygame.K_v:       self._toggle_pip,
            pygame.K_ESCAPE:  lambda: setattr(self, 'running', False),
        }
        f = actions.get(key)
        if f: f()

    def _toggle_wireframe(self):
        self.wireframe = not self.wireframe
        self.renderer.shape_renderer.wireframe = self.wireframe
        print(f">> Wireframe: {'ON' if self.wireframe else 'OFF'}")

    def _toggle_grid(self):
        self.show_grid = not self.show_grid
        self.config.set("rendering", "show_grid", value=self.show_grid)

    def _toggle_axes(self):
        self.show_axes = not self.show_axes
        self.config.set("rendering", "show_axes", value=self.show_axes)

    def _toggle_hud(self):
        self.show_hud = not self.show_hud
        self.config.set("rendering", "show_hud", value=self.show_hud)

    def _toggle_auto_rotate(self):
        self.auto_rotate = not self.auto_rotate
        print(f">> Auto-rotate: {'ON' if self.auto_rotate else 'OFF'}")

    def _toggle_guide(self):
        self.renderer.show_guide = not self.renderer.show_guide

    def _toggle_pip(self):
        self.pip_visible = not self.pip_visible

    def _cycle_color(self):
        self.color_idx = (self.color_idx + 1) % len(COLOR_PALETTE)
        name = COLOR_PALETTE[self.color_idx][0]
        print(f">> Color: {name}")

    def _cycle_bg(self):
        self.bg_theme_idx = (self.bg_theme_idx + 1) % len(BG_THEMES)
        self.renderer.set_background(self.bg_theme_idx)
        print(f">> Background theme: {self.bg_theme_idx + 1}")

    def _adjust_scale(self, delta):
        self.shape_scale = clamp(self.shape_scale + delta, 0.1, 5.0)

    def _take_screenshot(self):
        fname = screenshot(self.screen)
        print(f">> Screenshot: {fname}")

    def _save(self):
        color = list(COLOR_PALETTE[self.color_idx][1])
        save_blueprint(self.shapes[self.shape_idx], self.rot_x, self.rot_y,
                       self.zoom_level, color)

    def _load(self):
        bp = load_blueprint()
        if bp:
            self.shape_idx  = self._shape_idx(bp.get("shape", self.shapes[self.shape_idx]))
            rot = bp.get("rotation", {})
            self.rot_x = self.target_rot_x = rot.get("x", 0)
            self.rot_y = self.target_rot_y = rot.get("y", 0)
            self.zoom_level = bp.get("zoom_level", self.zoom_level)

    def _export(self):
        export_obj(self.shapes[self.shape_idx], size=self.shape_scale)

    # ------------------------------------------------------------------
    # Gesture processing
    # ------------------------------------------------------------------

    def process_gestures(self, data: dict):
        smoothing = self.config.get("controls", "smoothing_factor") or 0.12
        rot_sens  = self.config.get("controls", "rotation_sensitivity") or 0.5
        zoom_min  = self.config.get("camera", "zoom_min")
        zoom_max  = self.config.get("camera", "zoom_max")

        # ---- fist reset ----
        if data["fist_reset"]:
            self.rot_x = self.rot_y = self.target_rot_x = self.target_rot_y = 0.0
            self.zoom_level = self.config.get("camera", "initial_z")
            self.shape_scale = 1.0
            self._pinch_last_pos = None
            self._zoom_base_dist = None
            print(">> View reset!")

        hands = data["both_hands"]

        # ---- determine dominant mode ----
        if len(hands) == 2 and data["zoom_distance"] is not None:
            self._current_mode = GestureMode.ZOOM
        elif hands:
            self._current_mode = hands[0]["mode"]
        else:
            self._current_mode = GestureMode.NONE

        # ---- two-hand ZOOM ----
        if self._current_mode == GestureMode.ZOOM:
            dist = data["zoom_distance"]
            if self._zoom_base_dist is None:
                self._zoom_base_dist  = dist
                self._zoom_base_level = self.zoom_level
            ratio     = (dist / self._zoom_base_dist) if self._zoom_base_dist > 1 else 1.0
            target_z  = self._zoom_base_level * (1.0 / ratio)
            target_z  = clamp(target_z, zoom_min, zoom_max)
            self.zoom_level = lerp(self.zoom_level, target_z, smoothing * 2)
        else:
            self._zoom_base_dist = None

        # ---- single-hand modes ----
        if hands:
            h = hands[0]
            mode = h["mode"]

            # ROTATE — pinch + drag
            if mode == GestureMode.ROTATE and h["pinched"]:
                cx, cy = h["center"]
                if self._pinch_last_pos:
                    dx = cx - self._pinch_last_pos[0]
                    dy = cy - self._pinch_last_pos[1]
                    self.target_rot_y += dx * rot_sens
                    self.target_rot_x += dy * rot_sens
                self._pinch_last_pos = (cx, cy)
            else:
                if mode != GestureMode.ROTATE:
                    self._pinch_last_pos = None

            # PAN — move hand shifts zoom (simple pan approximation)
            if mode == GestureMode.PAN:
                cx, cy = h["center"]
                if self._pan_last:
                    dx = cx - self._pan_last[0]
                    dy = cy - self._pan_last[1]
                    self.target_rot_y += dx * rot_sens * 0.5
                    self.target_rot_x += dy * rot_sens * 0.5
                self._pan_last = (cx, cy)
            else:
                self._pan_last = None

            # COLOR — wave left/right to cycle
            if mode == GestureMode.COLOR and h["wave_dir"]:
                delta = +1 if h["wave_dir"] == "right" else -1
                self.color_idx = (self.color_idx + delta) % len(COLOR_PALETTE)
                print(f">> Color: {COLOR_PALETTE[self.color_idx][0]}")

            # SCALE — vertical hand movement
            if mode == GestureMode.SCALE:
                vd = h["vert_delta"]
                if abs(vd) > 2:
                    self.shape_scale = clamp(self.shape_scale + vd * 0.005, 0.1, 5.0)

            # SHAPE — wave left/right to cycle
            if mode == GestureMode.SHAPE and h["wave_dir"]:
                delta = +1 if h["wave_dir"] == "right" else -1
                self.shape_idx = (self.shape_idx + delta) % len(self.shapes)
                print(f">> Shape: {self.shapes[self.shape_idx]}")

        # ---- smooth rotations ----
        self.rot_x = lerp(self.rot_x, self.target_rot_x, smoothing)
        self.rot_y = lerp(self.rot_y, self.target_rot_y, smoothing)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        clock = pygame.time.Clock()
        webcam_frame = None

        while self.running:
            self.handle_events()

            # Auto-rotate
            if self.auto_rotate:
                self.target_rot_y += self.auto_rot_speed

            # Webcam + hand tracking
            data = {
                "hands_detected": 0, "left_hand": None, "right_hand": None,
                "both_hands": [],    "zoom_distance": None,
                "fist_reset": False, "fist_progress": 0.0,
            }
            if self.cap:
                ok, frame = self.cap.read()
                if ok:
                    webcam_frame, data = self.sensor.process_frame(frame)
                    self.process_gestures(data)

            # Render
            self.renderer.render_frame(
                screen       = self.screen,
                shape_type   = self.shapes[self.shape_idx],
                rot_x        = self.rot_x,
                rot_y        = self.rot_y,
                zoom_level   = self.zoom_level,
                shape_scale  = self.shape_scale,
                color_idx    = self.color_idx,
                current_mode = self._current_mode,
                webcam_frame = webcam_frame,
                pip_visible  = self.pip_visible,
                auto_rotate  = self.auto_rotate,
            )

            pygame.display.flip()
            clock.tick(60)

        self._cleanup()

    def _cleanup(self):
        if self.cap:
            self.cap.release()
        pygame.quit()
        print(">> SHUTDOWN COMPLETE.")


# ---------------------------------------------------------------------------

def main():
    try:
        app = VertexApp()
        app.run()
    except KeyboardInterrupt:
        print("\n>> Interrupted.")
    except Exception as e:
        import traceback
        print(f">> FATAL: {e}")
        traceback.print_exc()
    finally:
        try: pygame.quit()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass


if __name__ == "__main__":
    main()
