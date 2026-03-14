#!/bin/bash

# Project Vertex Run Script

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Print banner
echo ">> STARTING PROJECT VERTEX"
echo ">> ----------------------"

# Check if Python 3.11 is available
PYTHON_CMD="python3.11"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python3"
    echo ">> Checking python version..."
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ">> Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    echo ">> Installing dependencies..."
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check for model file
if [ ! -f "hand_landmarker.task" ]; then
    echo ">> Downloading MediaPipe hand landmarker model..."
    curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
fi

# Run the application
echo ">> Launching application..."
python main.py

# Deactivate on exit
deactivate
