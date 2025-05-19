# PosePlay: Motion Gaming Interface

## Project Overview
PosePlay is an interactive computer vision application that translates physical body movements into keyboard inputs for controlling web browser games such as Subway Surfers. Using TensorFlow and OpenCV, it captures the player's movements through a webcam and converts those movements into keypress events using pyautogui, allowing for hands-free gameplay.

Initially started, with code from [bahirul/subwaysuft-movenet](https://github.com/bahirul/subwaysuft-movenet). The original implementation was designed specifically for controlling Subway Surfers, through websites such as https://poki.com/en/g/subway-surfers with only using body movements. 

I've added hand gestures for power-ups, pausing or resuming the game, GUI to display current movements, and a background filter.
## Key Features
- **Real-time Pose Detection**: Uses Google's MoveNet Lightning model to detect 17 keypoints of the human body
- **Gesture Recognition**: Identifies movements like left/right movements, jumping, crouching, and hand raises
- **Background Replacement**: Removes your background and replaces it with a custom image
- **Visual Feedback**: Displays a skeleton overlay and movement guidelines to help users understand their detected movements
- **Streamlit Interface**: Simple GUI with toggles for background, skeleton, and guidelines

## Improvements Over Original Implementation
- Added Streamlit GUI for easier configuration and visualization
- Expanded gesture recognition capabilities (added "Power Up" and "Pause/Resume" gestures)
- Added background replacement using cvzone's SelfiSegmentation
- Enhanced visual feedback with toggleable display options
- Improved movement detection and reliability

## Usage Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Place a background image named `background.jpeg` in the same directory
3. Launch the application: `streamlit run poseplay.py`
4. Use the sidebar checkboxes to toggle:
   - Background replacement
   - Skeleton overlay
   - Movement guidelines
5. Control games with the following movements:
   - **Move Right**: Position your body toward the right side of the frame
   - **Move Left**: Position your body toward the left side of the frame
   - **Jump**: Raise your head above the upper guideline
   - **Crouch**: Lower your head below the lower guideline
   - **Power Up**: Raise one hand above shoulder level
   - **Pause/Resume**: Raise both hands above shoulder level

## Future Enhancements
- Additional gesture recognition capabilities
- Different filters based on the game that you're playing
- Performance optimizations for smoother gameplay

---
*This project was developed as part of CMSC 191 - Computer Vision in Python final project.*