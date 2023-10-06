# subwaysurf-movenet

This code is intended to automate keypress actions in the popular mobile game Subway Surfers using pose detection. It uses TensorFlow and OpenCV to capture the player's movements through a webcam and translates those movements into keypress events.

## Concept

The main concept of this code is detect keypoint for movement and then send or press keyboard key to control the game. The keypoint is detected using MoveNet Lightning model from TensorFlow Hub. The model is trained to detect 17 keypoints of human body. For more information about the model, you can read [here](https://tfhub.dev/google/movenet/singlepose/lightning/4).

## Features

- Detects the player's pose using the MoveNet Lightning model from TensorFlow Hub.
- Recognizes key movements such as running left, running right, jumping, and crouching.
- Automatically simulates corresponding keypress actions based on the detected movements.
- Allows the user to toggle keypress actions on and off using the spacebar.
- Provides real-time feedback on the detected movements and keypress actions through the webcam feed.
- Logs keypress and movement actions with timestamps for reference.

## Requirements

1. Python 3.6 or higher (Download from [here](https://www.python.org/downloads/))
2. TensorFlow 2.5.0 or higher (See usage section step 1 for installation)
3. TensorFlow Hub 0.12.0 or higher (See usage section step 1 for installation)
4. OpenCV 4.5.2 or higher (See usage section step 1 for installation)
5. NumPy 1.19.5 or higher (See usage section step 1 for installation)


## Usage

To use this code for automating keypress actions in Subway Surfers, follow these steps:

1. Make sure you have the required dependencies installed: TensorFlow, TensorFlow Hub, OpenCV, and NumPy.
- ```pip install -r requirements.txt```

2. Connect a webcam or ensure that your camera device is accessible by specifying the `--device` argument with the appropriate camera device index.

3. Customize the frame width and height using the `--width` and `--height` arguments if needed.

4. Run the script. The webcam feed will open, and the code will begin detecting your movements.

5. Use the following movements to control the game:
   - Right: Move your right hip to the right side of the frame.
   - Left: Move your left hip to the left side of the frame.
   - Jump: Jump until left shoulder and right shoulder on top frame.
   - Crouch: Lower your head below on the frame.

6. Press the spacebar to toggle keypress actions on and off.

7. To exit the script, press the "ESC" key.
