from pathlib import Path
import os
import cv2

# Telegram bot configuration
TOKEN = "YOUR_TOKEN_HERE"  # Telegram bot token for communication
TG_CHAT_ID = "YOUR_CHAT_ID"  # Chat ID for sending messages to a specific Telegram chat
UNKNOWN_TIMEOUT = 1  # Time interval (in minutes) to wait before considering a person unknown again

# Frame processing configuration
CHECK_INTERVAL = 2  # Interval (in seconds) to check for faces
FRAME_CHECK_RATE = 2  # How often to process frames (frame rate control)
FRAME_RESIZE_SCALE = 0.3  # Scale factor for resizing frames (used to speed up processing)

# Project paths
DEFAULT_PROJECT_PATH = Path(__file__).parent.parent  # Default path of the project directory
DEFAULT_ENCODINGS_PATH = Path(os.path.join(DEFAULT_PROJECT_PATH, "encodings.pkl"))  # Path for storing encodings
DEFAULT_FACES_PATH = Path(os.path.join(DEFAULT_PROJECT_PATH, "dataset"))  # Path to dataset of faces (images)
DEFAULT_MODEL = "hog"  # Default model for face detection ('hog' or 'cnn')
DEFAULT_IMAGE_SIZE = (640, 480)  # Default resolution for image processing

# Rectangle properties for face bounding box
RECTANGLE_COLOR = (0, 255, 0)  # Color of the rectangle (green, in BGR format)
RECTANGLE_THICKNESS = 2  # Thickness of the rectangle border

# Text properties for displaying information on the video frame
TEXT_COLOR = (0, 255, 0)  # Color of the text (green, in BGR format)
TEXT_FONT = cv2.FONT_HERSHEY_DUPLEX  # Font style for the text
TEXT_FONT_SIZE = 0.5  # Font size for the text
TEXT_THICKNESS = 1  # Thickness of the text to be drawn
