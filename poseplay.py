import cv2 as cv
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import streamlit as st
import pyautogui
import datetime
import time

# Faster pyautogui
pyautogui.PAUSE = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.FAILSAFE = False

def log_info(message):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{date_time}] {message}")

def run_inference(model, input_size, image):
    """
    Run inference on image to get keypoints and scores
    """
    image_height, image_width = image.shape[0], image.shape[1]

    # Preprocess image
    input_image = cv.resize(image, dsize=(input_size, input_size))  # Resize
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR to RGB
    input_image = cv.GaussianBlur(input_image, (5,5), 0)  # Gaussian blur
    input_image = input_image.reshape(-1, input_size, input_size, 3)
    input_image = tf.cast(input_image, dtype=tf.int32)  # Cast to int32
    
    # Run model
    outputs = model(input_image)

    keypoints_with_scores = outputs["output_0"].numpy()
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # Keypoints, Scores
    keypoints = []
    scores = []

    for idx in [i for i in range(17) if i not in [3,4]]:  # Skip left and right ear
        keypoint_x = int(keypoints_with_scores[idx][1] * image_width)
        keypoint_y = int(keypoints_with_scores[idx][0] * image_height)
        score = keypoints_with_scores[idx][2]

        keypoints.append((keypoint_x, keypoint_y))
        scores.append(score)

    return keypoints, scores

def draw_keypoints(image, keypoints, scores, keypoint_score_th, show_skeleton=True, show_guidelines=True):
    circle_color_outer = (255, 255, 255)  # White
    circle_color_inner = (0, 0, 255)  # Red
    line_skeleton_color = (0, 255, 0)  # Green
    
    # Create a copy of the image to draw on
    output_img = image.copy()
    
    # Draw skeleton lines if enabled
    if show_skeleton:
        # Eyes to nose
        for idx1, idx2 in [(1, 0), (2, 0)]:
            if scores[idx1] > keypoint_score_th and scores[idx2] > keypoint_score_th:
                cv.line(output_img, keypoints[idx1], keypoints[idx2], line_skeleton_color, 2)
        
        # Shoulders
        if scores[3] > keypoint_score_th and scores[4] > keypoint_score_th:
            cv.line(output_img, keypoints[3], keypoints[4], line_skeleton_color, 2)
        
        # Arms
        for idx1, idx2 in [(3, 5), (5, 7), (4, 6), (6, 8)]:
            if scores[idx1] > keypoint_score_th and scores[idx2] > keypoint_score_th:
                cv.line(output_img, keypoints[idx1], keypoints[idx2], line_skeleton_color, 2)
        
        # Torso
        for idx1, idx2 in [(3, 9), (4, 10), (9, 10)]:
            if scores[idx1] > keypoint_score_th and scores[idx2] > keypoint_score_th:
                cv.line(output_img, keypoints[idx1], keypoints[idx2], line_skeleton_color, 2)
        
        # Legs
        for idx1, idx2 in [(9, 11), (11, 13), (10, 12), (12, 14)]:
            if scores[idx1] > keypoint_score_th and scores[idx2] > keypoint_score_th:
                cv.line(output_img, keypoints[idx1], keypoints[idx2], line_skeleton_color, 2)

        # Draw keypoints
        for keypoint, score in zip(keypoints, scores):
            if score > keypoint_score_th:
                cv.circle(output_img, keypoint, 6, circle_color_outer, -1)  # White outer circle
                cv.circle(output_img, keypoint, 3, circle_color_inner, -1)  # Red inner circle

    # Logic for movement
    # Horizontal Movement
    if keypoints[4][0] > output_img.shape[1] / 2:
        movement = "Right"
    elif keypoints[3][0] < output_img.shape[1] / 2:
        movement = "Left"
    else:
        movement = "Standing"

    # Vertical Movement
    if keypoints[1][1] < output_img.shape[0] / 5 and keypoints[2][1] < output_img.shape[0] / 5:
        movement = "Jump"

    # Crouch Movement
    if keypoints[0][1] > output_img.shape[0] / 5 * 2:
        movement = "Crouch"

    # Special gestures
    if keypoints[8][1] < output_img.shape[0] / 5 * 2 and keypoints[7][1] < output_img.shape[0] / 5 * 2:
        movement = "Pause/Resume"
    elif keypoints[8][1] < output_img.shape[0] / 5 * 2 or keypoints[7][1] < output_img.shape[0] / 5 * 2:
        movement = "Power Up"

    # Draw guidelines if enabled
    if show_guidelines:
        # Jump line
        cv.line(output_img, (0, int(output_img.shape[0] / 5)), 
                (output_img.shape[1], int(output_img.shape[0] / 5)), (255, 255, 255), 2)
        # Crouch line
        cv.line(output_img, (0, int(output_img.shape[0] / 5 * 2)), 
                (output_img.shape[1], int(output_img.shape[0] / 5 * 2)), (255, 255, 255), 2)
        # Vertical Center line
        cv.line(output_img, (int(output_img.shape[1] / 2), 0), 
                (int(output_img.shape[1] / 2), output_img.shape[0]), (255, 255, 255), 2)

    return output_img, movement

def apply_background_replacement(frame, bg_image, segmentor):
    """
    Applies background replacement
    """
    if bg_image is not None:
        # Resize background to match frame dimensions
        resized_bg = cv.resize(bg_image, (frame.shape[1], frame.shape[0]))
        # Replace background
        processed_frame = segmentor.removeBG(frame, resized_bg, threshold=0.5)
        return processed_frame
    else:
        return frame

def handle_movement(frame_movement, last_movement):
    """
    Handles movement keypresses and logs.
    Returns the updated last_movement.
    """
    if frame_movement != last_movement:
        if frame_movement == "Right":
            pyautogui.press("right")
            log_info("Keypress: Right")
            last_movement = "Right"
            
        elif frame_movement == "Left":
            pyautogui.press("left")
            log_info("Keypress: Left")
            last_movement = "Left"
            
        elif frame_movement == "Jump":
            pyautogui.press("up")
            log_info("Keypress: Up")
            last_movement = "Jump"
            
        elif frame_movement == "Crouch":
            pyautogui.press("down")
            log_info("Keypress: Down")
            last_movement = "Crouch"
            
        elif frame_movement == "Standing":
            if last_movement == "Right":
                pyautogui.press("left")
                log_info("Keypress: Left")
            elif last_movement == "Left":
                pyautogui.press("right")
                log_info("Keypress: Right")
            last_movement = "Standing"
            
        elif frame_movement == "Pause/Resume":
            pyautogui.press("esc")
            log_info("Keypress: Pause/Resume")
            last_movement = "Pause/Resume"
            
        elif frame_movement == "Power Up":
            pyautogui.press("space")
            log_info("Keypress: Space (Power Up)")
            
        log_info(f"Movement: {frame_movement}")
    
    return last_movement

# Main Streamlit App
st.set_page_config(page_title="Poseplay Controller", layout="wide")
st.title("Poseplay Controller (Subway Surfer)")

# Set up the sidebar with toggle controls
st.sidebar.title("Settings")
threshold = st.sidebar.slider("Keypoint Score Threshold", 0.0, 1.0, 0.3, 0.01)
use_background = st.sidebar.checkbox("Show Background", value=True)
show_skeleton = st.sidebar.checkbox("Show Skeleton", value=True)
show_guidelines = st.sidebar.checkbox("Show Guidelines", value=True)

# Create layout
col1, col2 = st.columns([3, 1])

with col1:
    # Camera feed placeholder
    video_placeholder = st.empty()

with col2:
    # Stats display
    st.subheader("Game Controls")
    st.markdown("""
    Move your body to control Subway Surfers:
    
    - **Left/Right**: Move either shoulder across center line
    - **Jump**: Raise head above top line
    - **Crouch**: Lower head below bottom line
    - **Power Up**: Raise one hand
    - **Pause/Resume**: Raise both hands
    """)
    
    st.subheader("Current Status")
    movement_text = st.empty()
    count_text = st.empty()

# Load model
with st.spinner("Loading pose detection model..."):
    model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    model = hub.load(model_url).signatures["serving_default"]
    st.success("Model loaded successfully!")

# Initialize segmentation
segmentor = SelfiSegmentation()

# Load background image
try:
    bg_image = cv.imread("background.jpeg")
    if bg_image is not None:
        st.sidebar.success("Background image loaded!")
    else:
        st.sidebar.warning("No background image found. Using plain camera feed.")
        bg_image = None
except Exception as e:
    st.sidebar.error(f"Error loading background: {e}")
    bg_image = None

# Initialize camera
try:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to open camera!")
        st.stop()
except Exception as e:
    st.error(f"Camera error: {e}")
    st.stop()

# Initialize variables
last_movement = "Standing"
movement_count = 0
frame_count = 0
start_time = time.time()
fps = 0

# Main loop
try:
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to read from camera!")
            break
            
        # Flip frame
        frame = cv.flip(frame, 1)
        
        # Apply background replacement if enabled
        if use_background and bg_image is not None:
            processed_frame = apply_background_replacement(frame, bg_image, segmentor)
        else:
            processed_frame = frame
            
        # Run inference
        keypoints, scores = run_inference(model, 192, processed_frame) # 192 is the fixed input size for movenet
        
        # Draw visualization
        visualization, frame_movement = draw_keypoints(
            processed_frame, keypoints, scores, threshold, 
            show_skeleton=show_skeleton, 
            show_guidelines=show_guidelines
        )
            
        # Handle movement
        last_movement = handle_movement(frame_movement, last_movement)
            
        # Count movements
        if frame_movement != "Standing" and frame_movement != last_movement:
            movement_count += 1
            
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
            
        # Convert for display
        rgb_frame = cv.cvtColor(visualization, cv.COLOR_BGR2RGB)
        
        # Update UI
        video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        movement_text.markdown(f"**Current Movement:** {frame_movement}")
        count_text.markdown(f"**Movement Count:** {movement_count} | **FPS:** {fps:.1f}")
        
        # Small delay to prevent UI freezing
        time.sleep(0.01)
        
except Exception as e:
    st.error(f"Error: {e}")
finally:
    # Clean up
    if 'cap' in locals() and cap is not None:
        cap.release()