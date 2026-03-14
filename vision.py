"""
Vision processor — hand tracking and gesture recognition for Project Vertex.
Uses the GestureEngine for clean mode detection.
Returns annotated frames for PiP rendering (no cv2.imshow).
"""
import cv2
import mediapipe as mp
import math
import time
import os

from gesture_engine import GestureEngine, GestureMode, MODE_META


class HandSensor:
    def __init__(self, detection_con=0.5, track_con=0.5):
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.core import base_options as ba

        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        opts = HandLandmarkerOptions(
            base_options=ba.BaseOptions(model_asset_path=model_path),
            num_hands=2,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=track_con,
            min_tracking_confidence=track_con,
        )
        self.landmarker = HandLandmarker.create_from_options(opts)
        self.engine     = GestureEngine()

        # Drawing palette (BGR)
        self._cyan   = (255, 220,  0)
        self._white  = (255, 255, 255)
        self._orange = (  0, 160, 255)
        self._green  = (  0, 220,  0)
        self._red    = (  0,  0, 220)

        self.HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]

        # Per-hand state
        self.prev_centers: dict[str, tuple] = {}   # for velocity + swipe
        self.swipe_ref:    dict[str, tuple] = {}   # SHAPE/COLOR mode swipe origin

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process_frame(self, img):
        """Process one webcam frame.

        Returns:
            annotated_img : BGR frame with skeleton + gesture annotations drawn
            data          : dict with gesture state for main.py to consume
        """
        img  = cv2.flip(img, 1)
        h, w = img.shape[:2]

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect(mp_img)

        now  = time.time()
        data = {
            "hands_detected":   0,
            "left_hand":        None,
            "right_hand":       None,
            "both_hands":       [],        # list of hand dicts (up to 2)
            "zoom_distance":    None,      # pixel dist between index tips (2-hand)
            "fist_reset":       False,
            "fist_progress":    0.0,       # 0-1 hold progress for UI ring
        }

        if not result.hand_landmarks:
            return img, data

        data["hands_detected"] = len(result.hand_landmarks)
        all_hands = []

        for idx, (landmarks, handedness) in enumerate(
                zip(result.hand_landmarks, result.handedness)):

            label    = handedness[0].category_name   # "Left" / "Right"
            hand_id  = f"{label}_{idx}"

            # ---- pixel landmark list ----
            # Include 3D normalized coordinates (x, y, z) at indices 3, 4, 5
            lm_list = [[i, int(lm.x*w), int(lm.y*h), lm.x, lm.y, lm.z]
                       for i, lm in enumerate(landmarks)]

            # ---- gesture mode ----
            mode = self.engine.get_mode(hand_id, lm_list)

            # ---- fist reset ----
            if self.engine.check_fist_reset(hand_id, lm_list, now):
                data["fist_reset"] = True
            data["fist_progress"] = max(
                data["fist_progress"],
                self.engine.fist_hold_progress(hand_id, now))

            # ---- center + velocity ----
            cx = sum(p[1] for p in lm_list) // 21
            cy = sum(p[2] for p in lm_list) // 21
            prev = self.prev_centers.get(hand_id, (cx, cy))
            velocity = (cx - prev[0], cy - prev[1])
            self.prev_centers[hand_id] = (cx, cy)

            # ---- pinch ----
            ix, iy = lm_list[8][1], lm_list[8][2]
            tx, ty = lm_list[4][1], lm_list[4][2]
            pinch_dist = math.hypot(ix-tx, iy-ty)
            pinched    = pinch_dist < 42

            # ---- wave detection (for SHAPE / COLOR modes) ----
            wave_dir = None
            if mode in (GestureMode.SHAPE, GestureMode.COLOR):
                ref = self.swipe_ref.get(hand_id)
                if ref is None:
                    self.swipe_ref[hand_id] = (cx, cy)
                else:
                    dx = cx - ref[0]
                    if abs(dx) > 55:
                        wave_dir = "right" if dx > 0 else "left"
                        self.swipe_ref[hand_id] = (cx, cy)
            else:
                self.swipe_ref.pop(hand_id, None)

            # ---- vertical drag (for SCALE mode) ----
            vert_delta = 0
            if mode == GestureMode.SCALE:
                vert_delta = -velocity[1]   # up = positive

            hand_info = {
                "id":         hand_id,
                "label":      label,
                "mode":       mode,
                "center":     (cx, cy),
                "velocity":   velocity,
                "lm_list":    lm_list,
                "index_tip":  (ix, iy),
                "thumb_tip":  (tx, ty),
                "pinched":    pinched,
                "wave_dir":   wave_dir,
                "vert_delta": vert_delta,
            }
            all_hands.append(hand_info)

            if label == "Left":  data["left_hand"]  = hand_info
            else:                data["right_hand"] = hand_info

            # ---- draw skeleton ----
            self._draw_skeleton(img, landmarks, w, h)
            self._draw_gesture_label(img, mode, cx, cy, pinched,
                                     self.engine.fist_hold_progress(hand_id, now))

        data["both_hands"] = all_hands

        # ---- two-hand zoom distance ----
        if len(all_hands) == 2:
            p1 = all_hands[0]["index_tip"]
            p2 = all_hands[1]["index_tip"]
            dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
            data["zoom_distance"] = dist
            mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            cv2.line(img, p1, p2, self._green, 2)
            cv2.putText(img, "ZOOM", (mid[0]-22, mid[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self._green, 2)

        return img, data

    # ------------------------------------------------------------------
    # Drawing helpers (onto the cv2 frame)
    # ------------------------------------------------------------------

    def _draw_skeleton(self, img, landmarks, w, h):
        for a, b in self.HAND_CONNECTIONS:
            if a < len(landmarks) and b < len(landmarks):
                p1 = (int(landmarks[a].x*w), int(landmarks[a].y*h))
                p2 = (int(landmarks[b].x*w), int(landmarks[b].y*h))
                cv2.line(img, p1, p2, self._cyan, 1)
        for lm in landmarks:
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 3, self._white, cv2.FILLED)
            cv2.circle(img, (cx, cy), 3, self._cyan, 1)

    def _draw_gesture_label(self, img, mode, cx, cy, pinched, fist_prog):
        label, rgb = MODE_META[mode]
        bgr = (rgb[2], rgb[1], rgb[0])

        if mode == GestureMode.NONE and fist_prog > 0.05:
            # Draw hold-progress ring
            angle = int(360 * fist_prog)
            cv2.ellipse(img, (cx, cy), (28, 28), -90, 0, angle, self._orange, 3)
            cv2.putText(img, f"{int(fist_prog*100)}%",
                        (cx-20, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self._orange, 2)
        else:
            radius = 14 if mode == GestureMode.NONE else 18
            cv2.circle(img, (cx, cy), radius, bgr, 2)

        cv2.putText(img, label, (cx-45, cy-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, bgr, 2)

        if pinched:
            cv2.circle(img, (cx, cy), 10, self._red, cv2.FILLED)

    def get_distance(self, p1, p2):
        return math.hypot(p2[0]-p1[0], p2[1]-p1[1])
