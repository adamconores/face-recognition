from modules.config import *
from modules.utils import *
from pathlib import Path

import argparse
import asyncio
import cv2


async def main(args):
    """
    Main function to handle command-line arguments and execute corresponding functions.
    
    Parameters:
    - args: The command-line arguments parsed by the parser.
    """
    if args.train:
        # Train the model by processing the images and saving encodings
        await train()
    elif args.save:
        # Ensure 'person' and either 'folder' or 'image' are provided for saving encodings
        if args.person and (args.folder or args.image):
            if args.image:
                # Save encodings for a specific image
                await save(person=args.person, image=Path(args.image), fullpath=args.fullpath)
            else:
                # Save encodings for a folder of images
                await save(person=args.person, folder=Path(args.folder), fullpath=args.fullpath)
        else:
            # Error handling for missing arguments
            print("Error: 'person' and either 'folder' or 'image' must be provided to save encodings.")
    elif args.load:
        # Load and print the existing encodings
        encodings = await load()
        print(f"Loaded encodings: {encodings}")
    elif args.recognize:
        # Perform face recognition based on the provided input (image, folder, or video)
        if args.image or args.folder or args.video:
            unknown_encodings = []

            if args.image:
                # Process a single image for face recognition
                unknown_image = face_recognition.load_image_file(args.image)
                unknown_encodings = await get_face_encodings(unknown_image)
            
            elif args.folder:
                # Process all images in a specified folder
                folder_path = Path(args.folder)
                images = list(folder_path.glob("*"))
                for img_path in async_tqdm(images, desc="Processing folder", total=len(images)):
                    unknown_image = face_recognition.load_image_file(img_path)
                    encodings = await get_face_encodings(unknown_image)
                    unknown_encodings.extend(encodings)

            elif args.video:
                # Process real-time video stream for face recognition
                await process_video_stream()
        else:
            # Error handling for missing input
            print("Error: Either '--image' or '--folder' must be provided for recognition.")
    else:
        # Error message if no valid command is provided
        print("No valid command provided. Use --train, --save, --load, or --recognize.")


def parse_args():
    """
    Parse and return command-line arguments.
    
    Returns:
    - argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Face Recognition Training and Encoding")
    parser.add_argument('--train', action='store_true', help="Train the face encodings and save them")
    parser.add_argument('--save', action='store_true', help="Save face encodings for a person")
    parser.add_argument('--load', action='store_true', help="Load existing face encodings")
    parser.add_argument('--recognize', action='store_true', help="Recognize faces from an image, folder, or video")
    parser.add_argument('-p', '--person', type=str, help="Name of the person for saving encodings")
    parser.add_argument('-f', '--folder', type=str, help="Folder path for images to process")
    parser.add_argument('-i', '--image', type=str, help="Path to a single image for encoding or recognition")
    parser.add_argument('-v', '--video', action='store_true', help="Use video stream for real-time recognition")
    parser.add_argument('-fP', '--fullpath', action='store_true', help="Use full path for images in the folder")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments and run the main function asynchronously
    args = parse_args()
    asyncio.run(main(args))
