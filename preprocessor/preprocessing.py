import cv2
import numpy as np
from config import IMAGE_SIZE
import logging

def load_image(image_path):
    """
    Load an image from a path.

    Args:
        image_path: Path to the image file.

    Returns:
        numpy.ndarray: The loaded image as a numpy array.
    """
    print(image_path)
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from path: {image_path}")
        return image
    except Exception as e:
        raise ValueError(f"Error loading image from {image_path}: {e}")

def convert_to_rgb(image):
    """
    Convert an image to RGB format.

    Args:
        image: The input image (PIL.Image or numpy array).

    Returns:
        numpy.ndarray: The image in RGB format.
    """
    if isinstance(image, np.ndarray):
        # If it's already a numpy array, use it directly
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Convert PIL.Image to numpy array
        image = np.array(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def normalize_image(image):
    """
    Normalize image values to the range [0, 1].
    """
    return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

def resize_image(image, size=IMAGE_SIZE):
    """
    Resize image to the given dimensions.
    """
    return cv2.resize(image, size)

def extract_channels(image):
    """
    Extract RGB channels and grayscale from an image.
    """
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return red_channel, green_channel, blue_channel, grey_image

def apply_canny_edge_detection(grey_image, low_threshold=250, high_threshold=400):
    """
    Apply Canny edge detection on a grayscale image.
    """
    grey_border = cv2.Canny(grey_image, low_threshold, high_threshold)
    return cv2.convertScaleAbs(grey_border, alpha=(255.0))

def image_preprocessing(image_input, apply_canny=False):
    """
    Preprocess an image.

    Args:
        image_input: Path to the image or a preloaded image array.

    Returns:
        The preprocessed image.
    """
    # Call load_image only if the input is a string (file path)
    logging.info(f"Processing input as file path: {image_input}")
    logging.info(f"Reciving input of {type(image_input)} type")
    if isinstance(image_input, str):
        logging.info(f"Processing input as file path: {image_input}")
        image = load_image(image_input)
    else:
        logging.info(f"Processing input as numpy array. Shape: {image_input.shape}")
        image = image_input

    # Process the image
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    logging.info(f"Image shape: {image.shape}")
    image = convert_to_rgb(image)
    logging.info(f"Image converted to RGB")
    image = normalize_image(image)
    logging.info(f"Image normalized")
    image = resize_image(image, (224, 224))
    logging.info(f"Image resized to {image.shape}")
    if apply_canny: 
        logging.info(f"Applying Canny edge detection")
        red_channel, green_channel, blue_channel, grey_image = extract_channels(image)
        grey_image_8bit = cv2.convertScaleAbs(grey_image, alpha=(255.0))
        grey_border = apply_canny_edge_detection(grey_image_8bit)
        image = np.stack((red_channel, green_channel, blue_channel, grey_border), axis=-1)
    if len(image[image < 0]) > 0:
        logging.info(f"Negative values found in the image. Setting them to 0.")
        image[image < 0] = 0
    return image
