import cv2
import numpy as np
import os
from pdf2image import convert_from_path

import logging
def extract_images_from_pdf(pdf_path, output_dir,enumerator):
    """
    Extract each page of the PDF as an image.
    """
    images = convert_from_path(pdf_path)
    os.makedirs(output_dir, exist_ok=True)

    extracted_images = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{enumerator + i + 1}.png")
        image.save(image_path, "PNG")
        extracted_images.append(image_path)
    return extracted_images


def split_frames(image_path, output_dir, rows=3, cols=4, mode="grayscale", page_number=1, apply_clahe=True):
    """
    Function to split a grid-like image into frames and save them.
    """
    # Step 1: Read input image
    original_image = cv2.imread(image_path)
    if mode == "grayscale":
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
    elif mode == "hsv":
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        image = hsv_image[:, :, 2]
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
        hsv_image[..., 2] = image

    if original_image is None:
        print(f"Error: Image '{image_path}' not found.")
        return

    # Step 2: Preprocess the image (adaptive threshold + noise removal)
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 3: Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Filter and sort contours
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]  # Ignore small contours

    # Sort contours row-by-row
    def sort_key(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return (y // (image.shape[0] // rows)) * cols + (cols - (x // (image.shape[1] // cols)))

    contours = sorted(contours, key=sort_key)

    # Step 5: Crop and save each frame
    os.makedirs(output_dir, exist_ok=True)
    frame_number = 1
    page_prefix = f"page_{page_number}"
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the frame with a margin
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w += 2 * margin
        h += 2 * margin
        cropped_frame = original_image[y:y + h, x:x + w]
        if mode == "hsv" and apply_clahe:
            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cropped_frame[..., 2] = clahe.apply(cropped_frame[..., 2])
            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_HSV2BGR)

        # Save the cropped frame
        if w >= 100 and h >= 100:
            output_path = os.path.join(output_dir, f"{page_prefix}_frame_{frame_number}.png")
            cv2.imwrite(output_path, cropped_frame)
            frame_number += 1

    if(frame_number != 13):
        error_message = f"There is an error on page {page_number}. Expected 12 frames but got {frame_number - 1}."
        logging.error(error_message)
        print(error_message)
    print(f"Frames saved in: {output_dir}")

def rotate_saved_frames(output_dir):


    """
    Rotate all images in the directory by 90 degrees counterclockwise.
    """
    for frame in os.listdir(output_dir):
        frame_path = os.path.join(output_dir, frame)
        if os.path.isfile(frame_path) and frame.lower().endswith(".png"):
            image = cv2.imread(frame_path)
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(frame_path, rotated)

def create_video_from_frames(output_dir, video_path, frame_rate=24):
    frames = [os.path.join(output_dir, frame) for frame in os.listdir(output_dir) if frame.lower().endswith(".png")]

    frames.sort()  # Ensure frames are ordered correctly

    if not frames:
        print(f"No frames present in directory '{output_dir}' to create video.")
        return

    # Get the resolution from the first frame
    first_frame = cv2.imread(frames[0])
    height, width, _ = first_frame.shape
    # Set up VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

    for frame_path in frames:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved at: {video_path}")

logging.basicConfig(filename='errors.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
if __name__ == "__main__":
    import argparse
    from natsort import natsorted  # Added this import above main and any prior imports

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="org_pdfs", help="Directory containing PDF files to process.")
    parser.add_argument("--recursive", action="store_true", help="Enable recursive frame search.")
    args = parser.parse_args()
    # Input PDF and output directories
    temp_images_dir = "temp_pdf_images"
    final_output_dir = "cropped_frames"
    all_image_paths = []  # Placeholder for images extracted from all PDFs

    # Check if the directory exists and has PDF files
    if not os.path.isdir(args.pdf_dir):
        print(f"Error: Directory '{args.pdf_dir}' does not exist.")
        exit(1)
    pdf_files = natsorted([os.path.join(args.pdf_dir, f) for f in os.listdir(args.pdf_dir) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print(f"Error: No PDF files found in directory '{args.pdf_dir}'.")
        exit(1)

    # Step 1: Extract images from PDF
    print("Extracting images from PDFs...")
    page_offset = 0
    for pdf_file in pdf_files:
        print(f"Processing PDF: {pdf_file}")
        pdf_image_paths = extract_images_from_pdf(pdf_file, temp_images_dir,page_offset)
        all_image_paths.extend((path, idx + page_offset) for idx, path in enumerate(pdf_image_paths))
        page_offset += len(pdf_image_paths)

    # Step 2: Process each extracted image
    print("Processing individual frames...")
    for image_path, page_number in all_image_paths:
        split_frames(image_path, final_output_dir, page_number=page_number + 1)  # Sequential numbering across PDFs

    print("Processing complete. All frames saved!")
    print("Video creation complete!")
    print("Rotating frames 90 degrees counterclockwise...")
    rotate_saved_frames(final_output_dir)
    print("Frame rotation complete!")

    create_video_from_frames(final_output_dir, "output_video.mp4", frame_rate=1)
    print("Video creation complete!")
