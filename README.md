# Face Recognition Project

## Author

**Adam Conores**  
Google Account: [adamconores@gmail.com](mailto:adamconores@gmail.com)  
GitHub: [AdamConores](https://github.com/adamconores)

## Overview

This project is a Telegram bot that integrates real-time face recognition using webcam video streams. It can detect and recognize faces in real time and send notifications to Telegram with images of faces that are not previously known (unknown faces). The system can dynamically learn new faces by training on new images and save them for future recognition.

The core of the project relies on popular Python libraries such as `OpenCV`, `face_recognition`, and `aiogram` for asynchronous Telegram bot interactions. The bot uses face encodings to compare and identify faces, offering a real-time and reliable solution for face recognition tasks.

### Features:
- **Real-time face recognition**: Uses webcam or video stream to recognize faces live.
- **Unknown face detection**: When a new face is detected that is not in the known faces list, a photo is sent to Telegram.
- **Telegram notifications**: Sends images of unknown faces to a specified chat via Telegram bot.
- **Face encoding storage**: Save face encodings for known individuals and update them dynamically.
- **Asynchronous processing**: Uses `asyncio` and `aiofiles` for efficient asynchronous file and image processing.
- **Configurable model and display**: Customize rectangle color, text color, font, and the face recognition model.

## Installation

Follow the steps below to install and configure the project on your machine.

### 1. Clone the repository
```bash
git clone https://github.com/your-username/face-recognition-telegram-bot.git
cd face-recognition

2. Install dependencies
Install the required Python libraries using pip. This project uses a requirements.txt file which lists all the necessary dependencies.

bash
Copy code
pip install -r requirements.txt
3. Setup your Telegram Bot
You need a Telegram bot to interact with the system. To create a bot:

Open Telegram and search for the "BotFather".
Use /newbot to create a new bot and get your bot token.
Note down your bot's token and chat ID (for the chat where notifications will be sent).
Once you have your TOKEN and TG_CHAT_ID, set them in the config.py file.

4. Configuration
The config.py file contains all the configurable settings for the bot, including parameters for face recognition and Telegram notifications.

python
Copy code
# Telegram bot token
TOKEN = 'your-telegram-bot-token'

# Telegram chat ID to send notifications
TG_CHAT_ID = 'your-telegram-chat-id'

# Path to store encodings
DEFAULT_ENCODINGS_PATH = "encodings.pkl"

# Path to faces' images for training
DEFAULT_FACES_PATH = "faces"

# Default image size for resizing
DEFAULT_IMAGE_SIZE = (150, 150)

# Default recognition model ("hog" or "cnn")
DEFAULT_MODEL = "hog"

# Timeout for detecting same unknown face again
UNKNOWN_TIMEOUT = 5  # minutes

# Rectangle color and thickness for face bounding boxes
RECTANGLE_COLOR = (0, 255, 0)
RECTANGLE_THICKNESS = 2

# Text color, font size, and thickness for displaying names
TEXT_COLOR = (255, 255, 255)
TEXT_FONT = cv2.FONT_HERSHEY_DUPLEX
TEXT_FONT_SIZE = 0.6
TEXT_THICKNESS = 1
Usage
After setting up the bot and configuration file, you can start using the system.

1. Train the system with known faces
To train the system with known faces, you need to provide images of the people whose faces the system should recognize. Place these images in directories named after the person (e.g., faces/JohnDoe/). Then, you can run the train() function, which will load the images, extract face encodings, and store them in a file.

python
Copy code
await train()  # Call this to train the system
2. Start face recognition
Once the system is trained, you can start the face recognition process in real-time. The system will use the webcam or video stream to recognize faces.

python
Copy code
await process_video_stream()  # Starts the real-time face recognition
When an unknown face is detected, it will be saved and sent to the specified Telegram chat.
The bot will notify you with the image of the detected unknown face.
3. Save new face encodings
If you encounter a new person and wish to add their face to the recognition system, you can use the save() function to save their face encodings. You can save a single image or all images in a folder.

python
Copy code
# Save an individual image
await save(image=path_to_image, person="JohnDoe")

# Save all images in a folder
await save(folder=path_to_folder, person="JohnDoe")
4. Asynchronous face recognition
Face recognition and face saving operations are handled asynchronously to ensure efficient processing, especially in real-time video stream scenarios.

File Structure
Here's an overview of the file and folder structure:

bash
Copy code
/face-recognition-telegram-bot
│
├── /config.py               # Configuration file (API keys, parameters)
├── /main.py                 # Main script for running the bot and face recognition
├── /requirements.txt        # Python dependencies
├── /tmp                     # Temporary images (e.g., captured unknown faces)
├── /faces                   # Folder for storing images of known faces
└── /encodings.pkl           # Serialized file to store face encodings of known people
config.py: Contains all configurations, such as Telegram bot token, chat ID, and settings for face recognition.
main.py: Contains the main logic for face recognition, video streaming, and bot interaction.
faces/: Store images of known people here, each in their own subfolder.
tmp/: Temporary folder to save images (e.g., unknown faces detected by the system).
encodings.pkl: The file where face encodings of known people are saved. This file is loaded and updated during the training process.
Asynchronous Processing
This project uses Python's asyncio and aiofiles for asynchronous processing. This approach allows the bot to handle video streams and face recognition without blocking other tasks, such as saving images and interacting with the Telegram bot.

The face recognition, saving, and loading of face encodings are performed asynchronously to ensure smooth operation.
async_tqdm is used for displaying progress bars during batch processing (e.g., when training with multiple images).
