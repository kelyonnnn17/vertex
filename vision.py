import cv2
import mediapipe as mp
import math

class HandSensor:
    def __init__(self, detection_con=0.7, track_con=0.7):
        # 1. Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,     # We need 2 hands for Zoom/Measure
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 2. Define the "Stark Tech" Palette (BGR Colors)
        self.col_cyan = (255, 255, 0)
        self.col_white = (255, 255, 255)
        self.col_red = (0, 0, 255)
        self.col_green = (0, 255, 0)

        # 3. Tuning Variables
        self.PINCH_THRESHOLD = 40  # Distance in pixels to trigger a "Click"

    def get_distance(self, p1, p2):
        """Math helper: Calculate distance between two points (x,y)"""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def process_frame(self, img):
        """
        Takes a raw webcam frame.
        Returns:
        1. img: The frame with the skeleton drawn on it.
        2. data: A dictionary containing hand coordinates and states.
        """
        # Flip the image (Mirror effect) and convert to RGB
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        h, w, c = img.shape
        
        # Prepare the Data Package
        data = {
            "hands_detected": 0,
            "left_hand": None,   # Will store {center: (x,y), pinched: bool}
            "right_hand": None,
            "distance_between_hands": 0 # For Zoom/Measure
        }

        if results.multi_hand_landmarks:
            data["hands_detected"] = len(results.multi_hand_landmarks)
            temp_hands = []

            for hand_type, hand_lms in zip(results.multi_handedness, results.multi_hand_landmarks):
                # DRAW: The Skeleton
                self.mp_draw.draw_landmarks(
                    img, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=self.col_white, thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=self.col_cyan, thickness=1, circle_radius=1)
                )

                # LOGIC: Extract Key Coordinates
                lm_list = []
                for id, lm in enumerate(hand_lms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])

                # Get Thumb (4) and Index (8) positions
                idx_x, idx_y = lm_list[8][1], lm_list[8][2]
                thumb_x, thumb_y = lm_list[4][1], lm_list[4][2]
                
                # Calculate Center & Pinch State
                center_x, center_y = (idx_x + thumb_x) // 2, (idx_y + thumb_y) // 2
                pinch_dist = self.get_distance((idx_x, idx_y), (thumb_x, thumb_y))
                is_pinched = pinch_dist < self.PINCH_THRESHOLD

                # VISUAL: "Lock" Indicator
                if is_pinched:
                    cv2.circle(img, (center_x, center_y), 10, self.col_red, cv2.FILLED)
                    cv2.putText(img, "LOCK", (center_x-20, center_y-20), 
                                cv2.FONT_HERSHEY_PLAIN, 1, self.col_red, 2)

                # Store Info
                label = hand_type.classification[0].label
                hand_info = {
                    "type": label,
                    "index_tip": (idx_x, idx_y),
                    "center": (center_x, center_y),
                    "pinched": is_pinched
                }
                
                # Assign to correct slot (Note: Mirror flip swaps L/R labels sometimes)
                if label == "Left": data["left_hand"] = hand_info
                else: data["right_hand"] = hand_info
                
                temp_hands.append(hand_info)

            # LOGIC: Dual Hand Measurement (The "Caliper")
            if len(temp_hands) == 2:
                p1 = temp_hands[0]["index_tip"]
                p2 = temp_hands[1]["index_tip"]
                
                dist = self.get_distance(p1, p2)
                data["distance_between_hands"] = int(dist)
                
                # VISUAL: Draw the Tape Measure
                cv2.line(img, p1, p2, self.col_green, 2)
                mid_x, mid_y = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
                cv2.putText(img, f"{int(dist)} px", (mid_x, mid_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.col_green, 2)

        return img, data

# --- UNIT TEST ---
if __name__ == "__main__":
    # If we run this file directly, it launches a test window
    cap = cv2.VideoCapture(0)
    sensor = HandSensor()
    print(">> SENSOR ONLINE. Press 'q' to exit.")
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame, data = sensor.process_frame(frame)
        
        cv2.imshow("PROJECT VERTEX | SENSOR TEST", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()