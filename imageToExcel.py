import cv2  # OpenCV for image processing
import pytesseract  # OCR library
import openpyxl  # For creating Excel files
import numpy as np  # For array manipulation
import tkinter as tk
import math
from tkinter import filedialog
import easyocr
import matplotlib.pyplot as plt


def remove_grid_lines(image):
    """Detects and removes grid lines from a grayscale image."""
    if image is None:
        return None
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Hough Line Transform parameters:
    rho = 1             # Distance resolution in pixels
    theta = np.pi / 180 # Angle resolution in radians
    threshold = 400     # Minimum number of intersections to detect a line
    min_line_length = 500 # Minimum length of a line to be considered
    max_line_gap = 22  # Maximum allowed gap between line segments to join them

    lines = cv2.HoughLinesP(blurred, rho, theta, threshold, min_line_length, max_line_gap)
    if lines is None:
        return image
    mask = np.zeros_like(image)
    angle_tolerance = np.pi / 90
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(abs(y2 - y1), abs(x2 - x1))
        if angle < angle_tolerance or abs(angle - np.pi) < angle_tolerance or abs(angle - np.pi / 2) < angle_tolerance:
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
    if np.sum(mask) == 0:
        return image
    dilated_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    return cv2.inpaint(image, dilated_mask, 2, cv2.INPAINT_NS)


def extract_text_from_image(image_path):
    """Extracts text from the image after removing grid lines."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError("Error: Image not found")

        removed_lines_image = remove_grid_lines(img)

        if removed_lines_image is None:
            print("Warning: Grid line removal failed or image is empty.")
            processed_image = img  # Use the original grayscale image
        else:
            processed_image = removed_lines_image

        reader = easyocr.Reader(['en'], gpu=False)
        detection_result = reader.readtext(processed_image)
        word_list = [detect[1] for detect in detection_result]
        print(f"Extracted Text:\n---\n{word_list}\n---")

        original_color_image = cv2.imread(image_path)
        if original_color_image is not None:
            for bbox, text, confidence in detection_result:
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                cv2.rectangle(original_color_image, top_left, bottom_right, (0, 255, 0), 5)
            plt.imshow(cv2.cvtColor(removed_lines_image, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            print("Warning: Could not load color image for bounding boxes.")

        return word_list

    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        return []
    except Exception as e:
        print(f"Error during text extraction: {e}")
        return []


def structure_data(text):
    """Structures the extracted text into a list of lists."""
    lines = text.strip().split('\n')
    data = [list(filter(None, line.split(' '))) for line in lines]
    max_cols = max(len(row) for row in data) if data else 0
    for row in data:
        row.extend([""] * (max_cols - len(row)))
    return data


def create_excel_file(data):
    """Creates an Excel file from the given data."""
    if not data:
        print("Warning: No data to write to Excel file.")
        return
    excel_output_path = select_file_output()
    if not excel_output_path:
        return
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    for row_index, row in enumerate(data, start=1):
        for col_index, value in enumerate(row, start=1):
            sheet.cell(row=row_index, column=col_index, value=value)
    workbook.save(excel_output_path)
    print(f"Excel file created successfully at {excel_output_path}")


def select_file_input():
    """Opens a file dialog for image selection."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename()
    root.destroy()
    return path


def select_file_output():
    """Opens a file dialog for saving the Excel file."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.asksaveasfilename(
        title="Save File",
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")],
        initialdir="."
    )
    root.destroy()
    return path


def main(image_path):
    """Main function to process the image."""
    try:
        extracted_text = extract_text_from_image(image_path)
        '''
        if extracted_text:
            structured_data = structure_data('\n'.join(extracted_text))
            create_excel_file(structured_data)
        else:
            print("Error: Text extraction failed or no text found.")
         '''   
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    image_file_path = select_file_input()
    if image_file_path:
        main(image_file_path)