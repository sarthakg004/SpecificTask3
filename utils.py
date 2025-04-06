import pandas
import fitz
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import numpy as np
import pytesseract
import shutil
import numpy as np
from difflib import SequenceMatcher
import numpy as np


## Creating a fuction to convert pdf to images and split if double page
def pdf_to_images(pdf_path, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    pdf_doc = fitz.open(pdf_path)
    image_counter = 1 

    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        pix = page.get_pixmap()  # Render page as image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # If the page is significantly wider than tall, assume it's a double-page
        if img.width > img.height * 1.2:
            left = img.crop((0, 0, img.width // 2, img.height))  # Left half
            right = img.crop((img.width // 2, 0, img.width, img.height))  # Right half

            left.save(os.path.join(output_dir, f"{image_counter}.png"), format="PNG")
            image_counter += 1
            right.save(os.path.join(output_dir, f"{image_counter}.png"), format="PNG")
        else:
            img.save(os.path.join(output_dir, f"{image_counter}.png"), format="PNG")

        image_counter += 1  # Increment for next image

    print(f"Processing complete! Saved {image_counter - 1} images in '{output_dir}'.")
    
    

def display_sample(dir, sample_number=5):

    sample_path = os.path.join(dir, f"{sample_number}.png")

    if os.path.exists(sample_path):
        img = Image.open(sample_path)
        plt.figure(figsize=(4, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Sample Image: {sample_number}.png")
        plt.show()
    else:
        print(f"Image {sample_number}.png not found in '{dir}'.")
        
        
import os
import string
from docx import Document

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def extract_text_by_page(docx_file, output_dir):
    document = Document(docx_file)
    pages = []
    current_page = []
    start_reading = False  # Flag to track when to start reading

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if not text:  # Skip initial empty lines
            continue

        if (text == "****" or "pdf" in text.lower()) and not start_reading:
            start_reading = True  # Start reading after the first "****"
            continue

        if not start_reading:
            continue  # Skip lines before the first "****"

        if text == "****" or "pdf" in text.lower() or "end of extract" in text.lower() or "--this part intentionally left blank to check after test--" in text.lower():  # Page separator found
            if current_page:
                pages.append("\n".join(current_page))  # Save current page
                current_page = []  # Start a new page
        else:
            text = remove_punctuation(text)  # Remove punctuation
            current_page.append(text)

    # Append the last page if it's not empty
    if current_page:
        pages.append("\n".join(current_page))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save each page separately in its own text file
    for idx, page_text in enumerate(pages, start=1):
        page_file = os.path.join(output_dir, f"page_{idx}.txt")
        with open(page_file, "w", encoding="utf-8") as f:
            f.write(page_text)

    print(f"Extracted {len(pages)} pages and saved them in '{output_dir}'.")

def load_image(image_path):
    image = cv2.imread(image_path)  # Load image (default is BGR)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image

def convert_to_grayscale(image):
    """Converts an image to grayscale if it's not already."""
    if len(image.shape) == 3:  # Check if the image is in color (BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # Already grayscale

def correct_skew(image):
    """Corrects skew in an image, supporting both grayscale and RGB inputs."""
    # Store original color state
    is_color = len(image.shape) == 3
    
    # Create grayscale copy for skew detection
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No text found, skipping skew correction.")
        return image  # Return original image with 0° skew angle

    # Find the largest contour (assumed to be text block)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area bounding box
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    # Normalize the angle to be between -45° and +45°
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    # Get image dimensions and center
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation with border replication
    deskewed = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return deskewed

def normalize_image(image):
    # Check if the image is RGB
    if len(image.shape) == 3:
        r, g, b = cv2.split(image)
        r_norm = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
        g_norm = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
        normalized = cv2.merge([r_norm, g_norm, b_norm])
        return normalized
    else:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
def ensure_300ppi(image, target_dpi=300):
    
    height, width = image.shape[:2]

    # Assume A4 document size in inches (common for scanned books)
    a4_width = 8.27  # inches
    a4_height = 11.69  # inches

    dpi_x = width / a4_width
    dpi_y = height / a4_height
    
    # Convert to PIL image, preserving color if needed
    image_pil = Image.fromarray(image)

    if dpi_x < target_dpi or dpi_y < target_dpi:
        # Calculate upscale factor
        scale_factor = target_dpi / min(dpi_x, dpi_y)  # Scale based on the lower DPI

        # Compute new size
        new_size = (int(image_pil.width * scale_factor), int(image_pil.height * scale_factor))

        # Resize using high-quality Lanczos resampling
        image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)

        # Set the new DPI metadata
        image_pil.info['dpi'] = (target_dpi, target_dpi)

    # Convert back to OpenCV format
    image_upscaled = np.array(image_pil)

    return image_upscaled

def remove_bleed_dual_layer(image):
    """Removes bleed-through, supporting both grayscale and RGB inputs."""
    # Check if image is color
    is_color = len(image.shape) == 3
    
    if is_color:
        # Convert to LAB color space which separates luminance from color
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_image)
        
        # Apply processing to luminance channel
        kernel = np.ones((51, 51), np.uint8)
        background = cv2.dilate(l, kernel, iterations=2)
        background = cv2.medianBlur(background, 21)
        
        # Subtract background to get foreground
        l_processed = 255 - cv2.absdiff(l, background)
        
        # Normalize to improve contrast
        l_processed = cv2.normalize(l_processed, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Merge channels back
        result_lab = cv2.merge([l_processed, a, b])
        
        # Convert back to RGB
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
    else:
        # Original grayscale processing
        kernel = np.ones((51, 51), np.uint8)
        background = cv2.dilate(image, kernel, iterations=2)
        background = cv2.medianBlur(background, 21)
        
        # Subtract background to get foreground
        diff = 255 - cv2.absdiff(image, background)
        
        # Normalize to improve contrast
        result = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    return result

from skimage import restoration

def denoise_image(image, method="nlm"):

    # Check if input is valid
    if image is None or image.size == 0:
        print("Error: Empty input image")
        return None
    
    # Ensure image is in uint8 format for OpenCV functions
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Make a copy to avoid modifying the original
    image_copy = image.copy()
    is_color = len(image.shape) == 3
 
    if method == "bilateral":
        # Bilateral filter preserves edges while removing noise
        if is_color:
            return cv2.bilateralFilter(image_copy, 9, 75, 75)
        else:
            return cv2.bilateralFilter(image_copy, 9, 75, 75)
    
    elif method == "nlm":
        # Non-local means denoising
        if is_color:
            return cv2.fastNlMeansDenoisingColored(image_copy, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image_copy, None, 10, 7, 21)
    
    elif method == "wiener":
        # Wiener filter (needs float conversion)
        if is_color:
            # Process each channel separately
            channels = cv2.split(image_copy.astype(np.float32) / 255.0)
            restored_channels = []
            
            for channel in channels:
                restored = restoration.wiener(channel, psf=np.ones((3, 3)) / 9, balance=0.3)
                restored = np.clip(restored * 255, 0, 255).astype(np.uint8)
                restored_channels.append(restored)
            
            return cv2.merge(restored_channels)
        else:
            float_img = image_copy.astype(np.float32) / 255.0
            restored = restoration.wiener(float_img, psf=np.ones((3, 3)) / 9, balance=0.3)
            return np.clip(restored * 255, 0, 255).astype(np.uint8)
        
def sharpen_image(image, method="laplacian"):
    is_color = len(image.shape) == 3
    
    if method == "laplacian":
        if is_color:
            # Process each channel separately
            r, g, b = cv2.split(image)
            r_sharp = cv2.Laplacian(r, cv2.CV_8U)
            g_sharp = cv2.Laplacian(g, cv2.CV_8U)
            b_sharp = cv2.Laplacian(b, cv2.CV_8U)
            
            r_result = cv2.add(r, r_sharp)
            g_result = cv2.add(g, g_sharp)
            b_result = cv2.add(b, b_sharp)
            
            return cv2.merge([r_result, g_result, b_result])
        else:
            laplacian = cv2.Laplacian(image, cv2.CV_8U)
            return cv2.add(image, laplacian)
    
    elif method == "custom":
        # Custom sharpening kernel
        kernel = np.array([[0, -2, 0], 
                          [-2, 9, -2],
                          [0, -2, 0]])
        
        if is_color:
            r, g, b = cv2.split(image)
            r_sharp = cv2.filter2D(r, -1, kernel)
            g_sharp = cv2.filter2D(g, -1, kernel)
            b_sharp = cv2.filter2D(b, -1, kernel)
            return cv2.merge([r_sharp, g_sharp, b_sharp])
        else:
            return cv2.filter2D(image, -1, kernel)
    
    elif method == "unsharp_mask":
        # Unsharp masking - better for color images
        if is_color:
            # Convert to LAB to separate luminance
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b_channel = cv2.split(lab)
            
            # Apply unsharp mask to luminance only
            gaussian = cv2.GaussianBlur(l, (0, 0), 3.0)
            unsharp_mask = cv2.addWeighted(l, 1.5, gaussian, -0.5, 0)
            
            # Merge back
            result = cv2.merge([unsharp_mask, a, b_channel])
            return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale
            gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
            return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    return image

from skimage import exposure

def enhance_contrast(image, method="clahe"):
    """Enhances contrast in an image, supporting both grayscale and RGB inputs."""
    is_color = len(image.shape) == 3
    
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        if is_color:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel only
            l_enhanced = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale
            return clahe.apply(image)
    
    elif method == "adaptive_eq":
        if is_color:
            # Convert to HSV to separate value from hue
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            
            # Apply adaptive eq to value channel
            v_enhanced = exposure.equalize_adapthist(v, clip_limit=0.03) * 255
            v_enhanced = v_enhanced.astype(np.uint8)
            
            # Merge channels and convert back to RGB
            enhanced_hsv = cv2.merge([h, s, v_enhanced])
            return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
        else:
            # For grayscale
            result = exposure.equalize_adapthist(image, clip_limit=0.03) * 255
            return result.astype(np.uint8)
    
    elif method == "stretch":
        if is_color:
            # Apply to each channel with care to preserve color relationships
            r, g, b = cv2.split(image)
            
            # Get global percentiles for consistent scaling
            low = np.min([np.percentile(r, 2), np.percentile(g, 2), np.percentile(b, 2)])
            high = np.max([np.percentile(r, 98), np.percentile(g, 98), np.percentile(b, 98)])
            
            # Scale each channel with the same limits
            r_stretched = np.clip((r - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
            g_stretched = np.clip((g - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
            b_stretched = np.clip((b - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
            
            return cv2.merge([r_stretched, g_stretched, b_stretched])
        else:
            # For grayscale
            p2, p98 = np.percentile(image, (2, 98))
            result = exposure.rescale_intensity(image, in_range=(p2, p98))
            return result.astype(np.uint8)
    
    return image

def morphological_operations(image, operation, kernel_size=(2, 2), iterations=1):
    is_color = len(image.shape) == 3
    kernel = np.ones(kernel_size, np.uint8)
    
    if is_color:
        r, g, b = cv2.split(image)
        
        if operation == "open":
            r_processed = cv2.morphologyEx(r, cv2.MORPH_OPEN, kernel, iterations=iterations)
            g_processed = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel, iterations=iterations)
            b_processed = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == "close":
            r_processed = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            g_processed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            b_processed = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == "dilate":
            r_processed = cv2.dilate(r, kernel, iterations=iterations)
            g_processed = cv2.dilate(g, kernel, iterations=iterations)
            b_processed = cv2.dilate(b, kernel, iterations=iterations)
        elif operation == "erode":
            r_processed = cv2.erode(r, kernel, iterations=iterations)
            g_processed = cv2.erode(g, kernel, iterations=iterations)
            b_processed = cv2.erode(b, kernel, iterations=iterations)
        else:
            return image
        
        return cv2.merge([r_processed, g_processed, b_processed])
    else:
        if operation == "open":
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == "close":
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == "dilate":
            return cv2.dilate(image, kernel, iterations=iterations)
        elif operation == "erode":
            return cv2.erode(image, kernel, iterations=iterations)
    
    return image

def binarize_image(image, method="otsu"):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    if method == "otsu":
        # Otsu's method for global thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    elif method == "adaptive":
        # Adaptive thresholding - good for uneven illumination
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 15, 8)
    
    return binary

###     Applies a binary mask to a color image to improve text regions.

def apply_binary_mask(color_image, binary_mask):
    # Ensure binary mask is properly formatted
    if len(binary_mask.shape) == 3:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_RGB2GRAY)
    
    # Create a 3-channel mask for direct multiplication
    mask_3channel = cv2.merge([binary_mask, binary_mask, binary_mask])
    
    # Normalize mask to 0-1 range for multiplication
    mask_normalized = mask_3channel / 255.0
    
    # Apply mask to color image
    masked_image = (color_image * mask_normalized).astype(np.uint8)
    
    return masked_image

def add_fixed_padding(image, padding=(10,10,10,10), padding_value=255):
    """Adds fixed padding to the image, supporting both grayscale and RGB inputs."""
    top, bottom, left, right = padding
    
    # Check if image is color
    is_color = len(image.shape) == 3
    
    if is_color:
        padded_image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, 
            value=[padding_value, padding_value, padding_value]  # RGB white
        )
    else:
        # For grayscale
        padded_image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, 
            value=padding_value
        )
    
    return padded_image


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def similarity_score(a, b):
    """Calculate string similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def map_bounding_boxes_to_transcript(image_path, bbox_path, transcript_path, output_path, tesseract_model=1, similarity_threshold=0.5):
    """Maps bounding boxes to transcript for a single image file."""
    # Read transcript
    with open(transcript_path, 'r') as f:
        transcript = f.read().strip()

    # Split transcript into words
    transcript_lines = transcript.split('\n')
    transcript_words = []
    for line in transcript_lines:
        transcript_words.extend(line.strip().split())

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image from {image_path}")
        return 0, 0, 0

    # Read bounding boxes
    with open(bbox_path, 'r') as f:
        bounding_boxes = [line.strip() for line in f.readlines() if line.strip()]

    # Result mappings
    mappings = []
    total_bbox_count = len(bounding_boxes)

    # Process each bounding box
    for bbox in bounding_boxes:
        try:
            # Parse the bounding box coordinates
            coords = [float(c) for c in bbox.split(',')]

            # Make sure we have enough coordinates
            if len(coords) < 8:
                print(f"Warning: Invalid bounding box format: {bbox}")
                continue

            x_coords = [coords[0], coords[2], coords[4], coords[6]]
            y_coords = [coords[1], coords[3], coords[5], coords[7]]

            # Get the rectangular region (min/max coordinates)
            x_min, y_min = int(min(x_coords)), int(min(y_coords))
            x_max, y_max = int(max(x_coords)), int(max(y_coords))

            # Ensure coordinates are within image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)

            # Skip if region is too small
            if x_max - x_min < 5 or y_max - y_min < 5:
                print(f"Warning: Skipping small region: {bbox}")
                continue

            # Extract the region from the image
            roi = image[y_min:y_max, x_min:x_max]

            # Skip if ROI is empty
            if roi.size == 0:
                print(f"Warning: Empty region for bbox: {bbox}")
                continue

            # Use pytesseract to get the text in this bounding box
            detected_text = pytesseract.image_to_string(roi,
                                                       config=f'--psm 7 --oem {tesseract_model} -l spa').strip()

            # Skip if no text detected
            if not detected_text:
                continue

            # Find the closest matching word in the transcript
            max_similarity = 0
            best_match = None

            for word in transcript_words:
                sim = similarity_score(detected_text, word)
                if sim > max_similarity:
                    max_similarity = sim
                    best_match = word

            # Only consider it a match if similarity is above threshold
            if max_similarity > similarity_threshold and best_match:
                # Add to mappings and remove matched word from transcript to avoid duplicates
                mappings.append((best_match, bbox, max_similarity))
                transcript_words.remove(best_match)
        except Exception as e:
            print(f"Error processing bounding box {bbox}: {e}")

    # Write results to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for word, bbox, _ in mappings:
            f.write(f"{word}\t{bbox}\n")

    return len(transcript_words) + len(mappings), len(mappings), total_bbox_count


def process_all_files(image_dir, bbox_dir, transcript_dir, output_dir,tesseract_model=1, similarity_threshold=0.5):
    """Processes all files for which transcripts are available."""
    total_words = 0
    total_mapped = 0
    total_bb = 0
    files_processed = 0
    no_transcript_files = 0

    # Make sure the output and no_transcript directories exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if directories exist
    for dir_path in [image_dir, bbox_dir, transcript_dir]:
        if not os.path.exists(dir_path):
            print(f"Error: Directory not found: {dir_path}")
            return

    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    print(f"Processing files in {image_dir}...")
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]

        image_path = os.path.join(image_dir, image_file)
        transcript_path = os.path.join(transcript_dir, f"{base_name}.txt")
        bbox_path = os.path.join(bbox_dir, f"{base_name}.txt")
        output_path = os.path.join(output_dir, f"{base_name}_mapped.txt")

        # Check if transcript exists
        if not os.path.exists(transcript_path):
            no_transcript_files += 1
            continue

        # Check if bbox file exists
        if not os.path.exists(bbox_path):
            print(f"Warning: Bounding box file not found: {bbox_path}")
            continue

        # Process the files
        try:
            words_count, mapped_count, bbox_count = map_bounding_boxes_to_transcript(
                image_path, bbox_path, transcript_path, output_path,
                tesseract_model, similarity_threshold
            )

            total_words += words_count
            total_mapped += mapped_count
            total_bb += bbox_count
            files_processed += 1

            print(f"File {base_name}: Found {words_count} words in transcript file, mapped {mapped_count} of {bbox_count} bounding boxes")
        except Exception as e:
            print(f"Error processing file {base_name}: {e}")

    print(f"\nMapping complete. Model Used: {tesseract_model}. "
          f"Processed {files_processed} files. Total {total_words} transcript words. "
          f"Processed {total_bb} bounding boxes and successfully mapped {total_mapped} words.\n"
          f"{no_transcript_files} files had no transcript")



def extract_and_save_regions(image_path, aligned_text_file, output_dir, start_index=0):
    """Extract regions from image using bounding boxes and save them with sequential filenames."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return [], start_index

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(aligned_text_file):
        print(f"Warning: Alignment file not found {aligned_text_file}")
        return [], start_index  # Skip if alignment file is missing

    results = []
    current_index = start_index

    with open(aligned_text_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split("\t") for line in f if "\t" in line]

    for text, bbox_str in lines:

        output_filename = f"image{current_index}.png"
        output_path = os.path.join(output_dir, output_filename)

        bbox = list(map(float, bbox_str.split(',')))
        pts = np.array(bbox, dtype=np.int32).reshape((4, 2))

        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        # Check if bounding box has valid dimensions
        if x_min >= x_max or y_min >= y_max:
            print(f"Warning: Invalid bounding box dimensions: {x_min},{y_min},{x_max},{y_max} - skipping")
            continue

        # Ensure box is within image boundaries
        height, width = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)

        # Check again after clamping to image boundaries
        if x_max - x_min < 2 or y_max - y_min < 2:
            continue

        cropped_region = image[y_min:y_max, x_min:x_max]

        # Validate cropping
        if cropped_region.size == 0 or cropped_region.shape[0] == 0 or cropped_region.shape[1] == 0:
            print(f"Warning: Empty cropped region - skipping")
            continue

        # Save the cropped image
        cv2.imwrite(output_path, cropped_region)

        results.append((output_filename, text))
        current_index += 1


    return results, current_index



import os
import cv2
import numpy as np
from PIL import Image

def analyze_image_sizes(directory):
    """Analyze the minimum, maximum, and average sizes of all images in a directory."""
    # Lists to store dimensions and file sizes
    widths = []
    heights = []

    # Count valid images
    total_images = 0

    # Supported image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

    # Process each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Skip if not a file or not an image file
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            continue

        try:
            # Get image dimensions using PIL
            with Image.open(file_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
            total_images += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Calculate statistics
    if total_images > 0:
        # Dimensions
        min_width, max_width = min(widths), max(widths)
        min_height, max_height = min(heights), max(heights)
        avg_width = sum(widths) / total_images
        avg_height = sum(heights) / total_images

        # Print results
        print(f"Analyzed {total_images} images in {directory}")
        print(f"Dimensions (width x height):")
        print(f"  Minimum: {min_width} x {min_height} pixels")
        print(f"  Maximum: {max_width} x {max_height} pixels")
        print(f"  Average: {avg_width:.1f} x {avg_height:.1f} pixels")
    else:
        print(f"No valid images found in {directory}")



import os
import unicodedata

def normalize_text_file(input_path, output_path=None):

    # Define character normalization function
    def normalize_char(c):
        # # Rule 1: Convert uppercase to lowercase
        # # c = c.lower()

        # # Rule 2: Handle u and v interchangeably (choosing u as standard)
        # if c == 'v':
        #     c = 'u'

        # # Rule 3: Handle different types of 's'
        # if c == 'ſ':  # Long s
        #     c = 's'

        # # Rule 4: Ignore accents except ñ
        # if c not in ['ñ']:
        #     # Decompose the character and keep only the base letter
        #     normalized = ''.join(char for char in unicodedata.normalize('NFD', c)
        #                         if not unicodedata.combining(char))
        #     c = normalized

        # # Rule 7: Replace ç with z
        # if c == 'ç':
        #     c = 'z'

        return c

    # Set default output path if none provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_normalized{ext}"

    # Read the input file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        with open(input_path, 'r', encoding='latin-1') as f:
            text = f.read()

    # Normalize each character
    normalized_text = ''.join(normalize_char(c) for c in text)

    # Write the normalized text to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(normalized_text)

    return output_path

def resize_and_pad(img, target_height, target_width, output_dir=None):
    # Calculate aspect ratio
    aspect = img.width / img.height

    # Determine new dimensions that fit within target size while preserving aspect ratio
    if aspect > (target_width / target_height):  # wider than target
        new_width = target_width
        new_height = int(new_width / aspect)
    else:  # taller than target
        new_height = target_height
        new_width = int(new_height * aspect)

    # Resize using LANCZOS for high quality
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    # Create new white image of target size
    padded_img = Image.new("RGB", (target_width, target_height), color="white")

    # Center the resized image
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    padded_img.paste(resized_img, (paste_x, paste_y))

    # Save the padded image
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(img.filename))
        padded_img.save(output_path)

    return padded_img

