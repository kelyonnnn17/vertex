# PROJECT VERTEX v3.0

A hand-gesture controlled 3D visualization workbench built with Python, OpenGL, PyGame, and MediaPipe. 

Vertex v3.0 represents a massive complete overhaul of the application, transforming it from a dual-window experiment into a sleek, unified, highly capable Picture-in-Picture 3D workspace.

## Features

### Core Mechanics
- **Single Unified Workspace**: The physical webcam feed is natively rendered as a live Picture-in-Picture (PiP) OpenGL texture directly inside the 3D environment.
- **Robust 3D Hand Tracking**: Built on MediaPipe, enhanced with a custom state-machine `GestureEngine` that uses mathematical vector dot products for flawless finger counting, rendering the application completely immune to hand-tilt perspective distortions.
- **Dynamic HUD**: Real-time Heads-Up Display showing FPS, Shape, Color, Rotation, Zoom, Vertex counts, and a live-updating Gesture Guide panel. 
- **10 Distinct 3D Shapes**: Cube, Pyramid, Sphere, Cylinder, Torus, Octahedron, Cone, Diamond, Icosahedron, Torus Knot.
- **12-Color Palette**: Cycle through vibrant colors instantaneously using hand gestures.

## Gesture Controls

Your current mode changes depending on **how many fingers you are holding up**:

| Fingers | Action |
|:---:|:---|
| **Fist** (hold) | **Reset** camera and rotation (fills a radial loading bar on screen) |
| **1 Finger** (pinch) | **Orbit/Rotate** the shape freely in 3D space |
| **2 Fingers** | **Pan** the view horizontally/vertically |
| **3 Fingers** (wave) | **Cycle Colors** |
| **4 Fingers** | **Scale** the shape up or down by moving your hand vertically |
| **5 Fingers** (wave) | **Cycle Shapes** |
| **2 Hands** (open) | **Zoom** in and out by moving your hands apart or together |

## Keyboard Shortcuts

| Key | Action |
|:---:|:---|
| `1` to `0` | Instantly select Shape 1 through 10 |
| `C` | Cycle Colors |
| `R` | Toggle Auto-Rotation |
| `B` | Cycle Background Themes (5 available) |
| `+/-` | Scale shape up/down |
| `W` | Toggle Wireframe Mode |
| `G` | Toggle Floor Grid |
| `A` | Toggle Coordinate Axes |
| `V` | Toggle Webcam PiP visibility |
| `?` | Toggle Gesture Guide panel |
| `H` | Toggle Text HUD |
| `S` | Save current workspace state (`blueprint.json`) |
| `L` | Load saved workspace state |
| `E` | Export current shape to `.obj` file |
| `P` | Take a Screenshot (saves as `.png`) |
| `ESC` | Quit |

## Installation

1. **Create and activate virtual environment:**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download hand landmarker model** (if not already present):
```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Usage

Run the application:
```bash
./run.sh
# or
python main.py
```

## Project Structure

```text
vertex/
├── main.py              # Main application entry point & event loop
├── vision.py            # MediaPipe processing and annotated frame generation
├── gesture_engine.py    # Custom vector-math gesture state machine
├── shapes.py            # Geometric definitions & normal calculations for 10 shapes
├── renderer.py          # Unified OpenGL rendering (3D + 2D PiP + Text Caching)
├── config.py            # Configuration loading and saving logic
├── utils.py             # Screenshots, OBJ export, Blueprint state saving
├── config.json          # Persisted configuration
├── requirements.txt     # Python dependencies
└── hand_landmarker.task # Local ML model
```

## Configuration
The configuration is automatically persisted to `config.json` every time you close the application. It saves your last window size, preferred shapes, color palette indices, active background themes, and UI visibility options.

---

**PROJECT VERTEX** - Where gestures meet geometry
