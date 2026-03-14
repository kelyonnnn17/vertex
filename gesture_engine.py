"""
Gesture Engine — finger-count based mode state machine for Project Vertex.

Maps a hand's landmark list to one of the GestureMode values, and tracks
fist-hold timing so the reset gesture fires exactly once per hold.
"""
import time
from enum import Enum, auto


class GestureMode(Enum):
    NONE   = auto()   # fist or unrecognised
    ROTATE = auto()   # 1 finger  — index up,  pinch+drag to orbit
    PAN    = auto()   # 2 fingers — peace sign, move to pan
    COLOR  = auto()   # 3 fingers — wave left/right to cycle palette
    SCALE  = auto()   # 4 fingers — move up/down to rescale
    SHAPE  = auto()   # 5 fingers — open palm, wave to cycle shapes
    ZOOM   = auto()   # two-hand override (set externally)


# (display label, RGB color) for each mode
MODE_META = {
    GestureMode.NONE:   ("FIST — hold to RESET", (180, 180, 180)),
    GestureMode.ROTATE: ("1 FNG — ORBIT",         (  0, 220, 220)),
    GestureMode.PAN:    ("2 FNG — PAN",            (100, 220, 100)),
    GestureMode.COLOR:  ("3 FNG — COLOR",          (220, 100, 220)),
    GestureMode.SCALE:  ("4 FNG — SCALE",          (220, 180,  60)),
    GestureMode.SHAPE:  ("5 FNG — SHAPE",          ( 60, 160, 255)),
    GestureMode.ZOOM:   ("2 HANDS — ZOOM",         (255, 140,  30)),
}

# Full gesture guide text shown in the HUD panel
GESTURE_GUIDE = [
    ("Fist (hold 0.8s)", "Reset camera"),
    ("1 finger + pinch", "Orbit / Rotate"),
    ("2 fingers",        "Pan view"),
    ("3 fingers + wave", "Cycle colour"),
    ("4 fingers",        "Scale shape"),
    ("5 fingers + wave", "Cycle shape"),
    ("2 hands open",     "Zoom"),
]


class GestureEngine:
    """Pure-logic gesture recognizer — no drawing, no OpenCV."""

    FIST_HOLD_SECONDS = 0.8

    def __init__(self):
        self._hold_gesture: dict[str, str]   = {}
        self._hold_start:   dict[str, float] = {}
        self._mode_history: dict[str, list]  = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_mode(self, hand_id: str, lm_list: list) -> "GestureMode":
        """Return the smoothed GestureMode for a single hand."""
        n = self.count_extended_fingers(lm_list)
        raw_mode = _FINGER_MODE_MAP.get(n, GestureMode.NONE)

        hist = self._mode_history.setdefault(hand_id, [])
        hist.append(raw_mode)
        if len(hist) > 7:
            hist.pop(0)

        # Return the most common mode in the recent history
        from collections import Counter
        most_common = Counter(hist).most_common(1)[0][0]
        return most_common

    def check_fist_reset(self, hand_id: str, lm_list: list, now: float) -> bool:
        """Return True exactly once when fist held >= FIST_HOLD_SECONDS.
        Resets internal timer so it won't re-fire until fist is released."""
        is_fist = self.count_extended_fingers(lm_list) == 0

        if is_fist:
            if self._hold_gesture.get(hand_id) == "fist":
                elapsed = now - self._hold_start.get(hand_id, now)
                if elapsed >= self.FIST_HOLD_SECONDS:
                    # Push start far into future so we don't fire again
                    self._hold_start[hand_id] = now + 9_999
                    return True
            else:
                self._hold_gesture[hand_id] = "fist"
                self._hold_start[hand_id] = now
        else:
            self._hold_gesture.pop(hand_id, None)
            self._hold_start.pop(hand_id, None)

        return False

    def fist_hold_progress(self, hand_id: str, now: float) -> float:
        """0.0→1.0 progress toward fist-reset (for the progress ring in UI)."""
        if self._hold_gesture.get(hand_id) != "fist":
            return 0.0
        elapsed = now - self._hold_start.get(hand_id, now)
        return min(elapsed / self.FIST_HOLD_SECONDS, 1.0)

    # ------------------------------------------------------------------
    # Finger counting
    # ------------------------------------------------------------------

    def count_extended_fingers(self, lm_list: list) -> int:
        """Count extended fingers (0-5) from a 21-point MediaPipe landmark list.

        Uses robust vector mathematics. A finger is extended if it points
        away from the palm (dot product > 0). This is immune to perspective
        distortion, foreshortening, and rotation.
        """
        if not lm_list or len(lm_list) < 21:
            return 0

        def pt(i):
            return (lm_list[i][1], lm_list[i][2])
            
        def dist(p1, p2):
            return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
            
        def sub(p1, p2):
            return (p1[0]-p2[0], p1[1]-p2[1])
            
        def dot(v1, v2):
            return v1[0]*v2[0] + v1[1]*v2[1]

        count = 0

        # ---- THUMB ----
        # Thumb is extended if its TIP (4) is further from the Pinky Base (17) 
        # than its middle joint (3) is.
        if dist(pt(4), pt(17)) > dist(pt(3), pt(17)):
            count += 1

        # ---- FOUR FINGERS ----
        # A finger is extended if the vector from its PIP to its TIP points in the 
        # same general direction as the vector from the WRIST to its MCP.
        # When curled, the TIP points back towards the wrist, making the dot product negative.
        fingers = [
            (5, 6, 8),   # Index:  MCP=5, PIP=6, TIP=8
            (9, 10, 12), # Middle: MCP=9, PIP=10, TIP=12
            (13, 14, 16),# Ring:   MCP=13, PIP=14, TIP=16
            (17, 18, 20) # Pinky:  MCP=17, PIP=18, TIP=20
        ]
        
        for mcp, pip, tip in fingers:
            palm_vec = sub(pt(mcp), pt(0))
            tip_vec  = sub(pt(tip), pt(pip))
            
            # If the vectors are in the same hemisphere, the finger is relatively straight.
            if dot(palm_vec, tip_vec) > 0:
                count += 1

        return count

    def get_hand_center(self, lm_list: list) -> tuple[int, int]:
        """Return the (x, y) centroid of all landmarks."""
        if not lm_list:
            return (0, 0)
        xs = [p[1] for p in lm_list]
        ys = [p[2] for p in lm_list]
        return (sum(xs) // len(xs), sum(ys) // len(ys))


# Lookup table: finger count → mode
_FINGER_MODE_MAP = {
    0: GestureMode.NONE,
    1: GestureMode.ROTATE,
    2: GestureMode.PAN,
    3: GestureMode.COLOR,
    4: GestureMode.SCALE,
    5: GestureMode.SHAPE,
}
