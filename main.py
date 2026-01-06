import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
import json
import os

# Import the Sensor we just built
from vision import HandSensor

# --- 3D ASSETS (The Wireframes) ---
vertices_cube = (
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
)
edges_cube = (
    (0,1), (0,3), (0,4), (2,1), (2,3), (2,7),
    (6,3), (6,4), (6,7), (5,1), (5,4), (5,7)
)

vertices_pyramid = (
    (0, 1, 0),   # Tip
    (-1, -1, 1), (1, -1, 1), (1, -1, -1), (-1, -1, -1) # Base
)
edges_pyramid = (
    (0,1), (0,2), (0,3), (0,4), # Sides
    (1,2), (2,3), (3,4), (4,1)  # Base
)

def draw_shape(shape_type):
    glBegin(GL_LINES)
    # COLOR: The Stark Cyan (R, G, B)
    glColor3f(0.0, 1.0, 1.0) 
    
    if shape_type == "cube":
        for edge in edges_cube:
            for vertex in edge:
                glVertex3fv(vertices_cube[vertex])
    elif shape_type == "pyramid":
        for edge in edges_pyramid:
            for vertex in edge:
                glVertex3fv(vertices_pyramid[vertex])
    glEnd()

def save_blueprint(shape, rot_x, rot_y, zoom):
    data = {
        "project_name": "VERTEX_V1",
        "shape": shape,
        "rotation": {"x": round(rot_x, 2), "y": round(rot_y, 2)},
        "zoom_level": round(zoom, 2)
    }
    with open("blueprint.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f">> 💾 SAVED BLUEPRINT: {data}")

def main():
    # 1. Initialize Window
    pygame.init()
    display = (1000, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("PROJECT VERTEX | WORKBENCH")

    # 2. Camera Setup
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5) # Start camera back 5 units

    # 3. Initialize The Eye
    sensor = HandSensor()
    cap = cv2.VideoCapture(0)

    # 4. Physics Variables
    rot_x, rot_y = 0, 0
    target_rot_x, target_rot_y = 0, 0
    zoom_level = -5
    current_shape = "cube"
    
    print(">> ENGINE ONLINE.")
    print(">> CONTROLS: [1] Cube, [2] Pyramid, [S] Save")

    while True:
        # --- A. KEYBOARD INPUTS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: current_shape = "cube"
                if event.key == pygame.K_2: current_shape = "pyramid"
                if event.key == pygame.K_s: save_blueprint(current_shape, rot_x, rot_y, zoom_level)

        # --- B. SENSOR INPUTS ---
        success, frame = cap.read()
        if not success: break
        
        # Get data from our vision.py script
        processed_frame, data = sensor.process_frame(frame)
        
        # Show the "HUD" (Webcam view) in a separate window
        cv2.imshow("PROJECT VERTEX | SENSOR HUD", processed_frame)

        # --- C. THE NEURAL BRIDGE (Logic) ---
        
        # MODE: DUAL HAND ZOOM (The Caliper)
        if data["hands_detected"] == 2:
            dist = data["distance_between_hands"]
            
            # Math: Map pixel distance (50px-400px) to Z-depth (-15 to -2)
            target_zoom = -15 + (dist / 25)
            if target_zoom > -2: target_zoom = -2 # Cap max zoom
            
            # Smooth the zoom (Linear Interpolation)
            zoom_level += (target_zoom - zoom_level) * 0.1

        # MODE: SINGLE HAND ROTATE (The Grab)
        elif data["hands_detected"] == 1:
            # Find which hand is active
            hand = data["left_hand"] or data["right_hand"]
            
            if hand and hand["pinched"]:
                cx, cy = hand["center"]
                
                # Math: Map screen position to rotation angle
                # (cx - 320) centers 0 rotation at the middle of the webcam
                target_rot_y = (cx - 320) * 0.5
                target_rot_x = (cy - 240) * 0.5

        # Physics: Apply "Momentum" (Smoothly drift to target)
        rot_x += (target_rot_x - rot_x) * 0.1
        rot_y += (target_rot_y - rot_y) * 0.1

        # --- D. RENDER CYCLE ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glLoadIdentity()
        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
        
        # Apply Transformations
        glTranslatef(0.0, 0.0, zoom_level) 
        glRotatef(rot_x, 1, 0, 0) # Rotate X axis
        glRotatef(rot_y, 0, 1, 0) # Rotate Y axis
        
        draw_shape(current_shape)

        pygame.display.flip()
        pygame.time.wait(10) # Cap FPS at ~60

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()