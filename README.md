# PosePlay: Motion Gaming Interface

## Original Source Attribution
This project is based on code from [bahirul/subwaysuft-movenet](https://github.com/bahirul/subwaysuft-movenet). The original implementation was designed specifically for controlling Subway Surfers, through websites such as https://poki.com/en/g/subway-surfers with only using body movements. 

I've added hand gestures for power-ups, pausing or resuming the game, GUI to display current movements, and a filter.

## Project Overview
PosePlay is an interactive computer vision application that translates physical body movements into keyboard inputs for controlling web browser games such as Subway Surfers. Using TensorFlow and OpenCV, it captures the player's movements through a webcam and converts those movements into keypress events using pyautogui, allowing for hands-free gameplay.

## Key Features
- **Real-time Pose Detection**: Uses Google's MoveNet Lightning model to detect 17 keypoints of the human body
- **Gesture Recognition**: Identifies movements like left/right movements, jumping, crouching, and hand raises
- **Customizable Controls**: Maps physical movements to keyboard actions for compatible games
- **Visual Feedback**: Displays a skeleton overlay to help users understand their detected movements
- **Configurable Settings**: Adjust sensitivity parameters and toggle functionality as needed

## Improvements Over Original Implementation
- Added GUI for easier configuration and visualization
- Expanded gesture recognition capabilities
- Enhanced visual feedback with customizable display options

## Technical Requirements
- Python 3.6 or higher
- TensorFlow 2.5.0 or higher
- TensorFlow Hub 0.12.0 or higher
- OpenCV 4.5.2 or higher
- NumPy 1.19.5 or higher
- PyAutoGUI for keyboard control
- [GUI framework: Tkinter/PyQt/etc.]

## Usage Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Launch the application: `python poseplay.py`
3. Use the GUI to configure your camera and control preferences
4. Stand in front of your camera where your full body is visible
5. Control games with the following movements:
   - **Move Right**: Position your body toward the right side of the frame
   - **Move Left**: Position your body toward the left side of the frame
   - **Jump**: Raise your shoulders to the upper portion of the frame
   - **Crouch**: Lower your head to the lower portion of the frame
   - **Action Button**: Raise either hand above shoulder level
   - **Pause/Resume** : Raise both hands above shoulder level
6. Press ESC to exit or use the GUI controls to stop the application

## Future Enhancements
- Additional gesture recognition capabilities
- Different filters based on game that you're playing

## Acknowledgments
- Original implementation by [bahirul](https://github.com/bahirul)
- MoveNet model from [TensorFlow Hub](https://tfhub.dev/google/movenet/singlepose/lightning/4)

## License
[Include appropriate license information]

---

*This project was developed as part of CMSC 191 - Computer Vision in Python final project.*