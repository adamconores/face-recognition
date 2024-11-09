from tqdm.asyncio import tqdm as async_tqdm
from collections import Counter
from aiogram import Bot, types
from datetime import datetime 
from .config import *

import face_recognition
import aiofiles
import asyncio
import pickle
import cv2
import numpy as np


bot = Bot(token=TOKEN)


async def process_video_stream() -> None:
    """Process real-time video stream and recognize faces."""
    target = True
    recent_time = datetime.now().minute

    # Initialize video capture
    video_capture = cv2.VideoCapture(0)

    # Process video frames in real-time
    while True:
        ret, d_frame = video_capture.read()
        if not ret:
            break
        frame = d_frame.copy()
        
        # Find face locations and encodings in the current frame
        face_locations = await get_face_locations(frame)
        face_encodings = await get_face_encodings(frame)

        # Initialize a list for face names
        face_names = []

        # Compare faces with known encodings
        for encoding in face_encodings:
            # Recognize each face asynchronously
            name = await recognize_face(encoding)
            if not target and datetime.now().minute - recent_time > UNKNOWN_TIMEOUT:
                print("yes")
                target = True
                recent_time = datetime.now().minute

            if name == "Unknown" and target:
                target = False
                cv2.imwrite("tmp_unknown.jpg", frame)
                await bot.send_photo(chat_id=TG_CHAT_ID, photo=types.FSInputFile("tmp_unknown.jpg"))
                os.remove("tmp_unknown.jpg")

            face_names.append(name)

        # Display the results on the original frame
        for face_location, name in zip(face_locations, face_names):
            frame = await display_face(frame, [face_location], name)

        # Show the frame
        cv2.imshow('Video', frame)

        # Capture image or exit conditions
        if cv2.waitKey(1) & 0xFF == ord('q'):
            img_name = input("Enter image name(leave blank if you won't): ")
            if img_name:
                cv2.imwrite(f"tmp/{img_name}.jpg", frame)
            break

        # if cv2.waitKey(1) & 0xFF == ord('s'):
        #     img_name = input("Enter image name: ")
        #     cv2.imwrite(f"tmp/{img_name}.jpg", d_frame) if img_name else cv2.imwrite(f"tmp/{datetime.now().now()}.jpg", d_frame)
        #     break
        # # Break loop with 'q' key

    # Release video capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()


async def resize_face(image: np.ndarray, 
                      size: tuple = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Resize face image to the specified size."""
    print(type(image))
    return cv2.resize(image, size)


async def train() -> None:
    """Train face encodings and save them asynchronously to DEFAULT_ENCODINGS_PATH."""
    known_faces = {}

    # Asynchronously process images in each folder
    for folder in DEFAULT_FACES_PATH.glob("*"):
        print(f"\nProcessing folder: {folder.name}...")
        name = folder.name
        known_faces[name] = []

        # Load images and retrieve encodings
        count = len(list(folder.glob("*")))
        async for image_path in async_tqdm(folder.glob("*"), total=count):
            image = face_recognition.load_image_file(image_path)
            encoding = await get_face_encodings(image)  # async call directly
            if encoding:
                known_faces[name].append(encoding[0])

    # Save encodings to file
    async with aiofiles.open(DEFAULT_ENCODINGS_PATH, "wb") as f:
        await f.write(pickle.dumps(known_faces))
    print("All encodings have been saved.")


async def load() -> dict:
    """Asynchronously load encodings from file."""
    if os.path.exists(DEFAULT_ENCODINGS_PATH):
        async with aiofiles.open(DEFAULT_ENCODINGS_PATH, "rb") as f:
            content = await f.read()
            return pickle.loads(content)
    return {}


async def save(image: Path = None, 
               person: str = "", 
               folder: Path = None, 
               fullpath: bool = False) -> None:
    """Asynchronously save new face encodings."""
    encodings = await load()  # First, asynchronously load existing encodings

    # Function to process image and get encoding
    async def process_image(img_path: Path) -> None:
        image = await resize_face(face_recognition.load_image_file(img_path))
        encoding = await get_face_encodings(image)  # await async call directly
        if encoding:
            encodings[person].append(encoding[0])

    # Create a new list if encoding doesn't exist
    if person not in encodings:
        encodings[person] = []

    # Save a single image encoding
    if image:
        await process_image(image)

    # Save all images in the specified folder
    if folder:
        folder_path = folder if fullpath else DEFAULT_PROJECT_PATH / folder.name
        images = list(folder_path.glob("*"))
        async for img_path in async_tqdm(images, desc=f"Processing {person} folder", total=len(images)):
            await process_image(img_path)

    # Update file after adding new encodings
    async with aiofiles.open(DEFAULT_ENCODINGS_PATH, "wb") as f:
        await f.write(pickle.dumps(encodings))
    print(f"New encodings for {person} have been saved.")


async def display_face(image: np.ndarray, 
                       face_locations: list, name: str) -> np.ndarray:
    """
    Display the image with rectangles drawn around faces and names labeled below,
    using configurable parameters for rectangle and text properties.
    
    Args:
    - image: The image on which to draw rectangles and names.
    - face_locations: List of tuples (top, right, bottom, left) for each face location.
    - names: List of names corresponding to each face in face_locations.
    """
    (top, right, bottom, left) = face_locations[0] if face_locations else []
    
    # Draw a rectangle around the face using config from config.py
    cv2.rectangle(image, (left, top), (right, bottom), RECTANGLE_COLOR, RECTANGLE_THICKNESS)

    # Draw the name below the rectangle using config from config.py
    cv2.putText(image, name, (left, bottom + 20), TEXT_FONT, TEXT_FONT_SIZE, TEXT_COLOR, TEXT_THICKNESS)

    return image


async def get_face_locations(image: np.ndarray, 
                             model: str = "hog") -> list:
    """Return the locations of faces in the image."""
    return face_recognition.face_locations(image, model=model)


async def get_face_encodings(image: np.ndarray) -> list:
    """Return the encodings of faces in the image."""
    return face_recognition.face_encodings(image, await get_face_locations(image, model=DEFAULT_MODEL))


async def recognize_face(unknown_encodings: list) -> dict:
    """
    Asynchronously recognize faces by comparing them with known encodings.
    Returns a dictionary where keys are person's names and values are the result of the match.
    """
    encodings = await load()

    async def compare_encodings(person_encodings: list) -> bool:
        """Compare person's encodings to unknown_encodings and return if they match."""
        matches = Counter(face_recognition.compare_faces(person_encodings, unknown_encodings))
        return matches.most_common(1)[0][0]

    # Use asyncio.gather to process each person in parallel
    tasks = []
    for (person, person_encodings) in encodings.items():
        tasks.append(compare_encodings(person_encodings))

    # Run the tasks concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Store the results in the dictionary
    for person, result in zip(encodings.keys(), results):
        if result:
            return person

    return "Unknown"
