# image_utils/bounding_box.py
import cv2
import numpy as np

def find_saliency_bounding_box(saliency_map, tolerance=0):
    """
    Find the bounding box from the saliency map with an optional tolerance.
    
    :param saliency_map: The saliency map from which to find the bounding box.
    :param tolerance: Number of pixels to expand the bounding box in all directions.
    :return: A tuple (x, y, w, h) representing the bounding box.
    """
    # Convert to numpy array and apply threshold
    saliency_np = (saliency_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
    saliency_np = cv2.threshold(saliency_np, 128, 255, cv2.THRESH_BINARY)[1]

    # Optionally apply blur
    saliency_np = cv2.GaussianBlur(saliency_np, (5, 5), 0)

    # Find contours and the bounding box
    contours, _ = cv2.findContours(saliency_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    for cnt in contours[1:]:
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        x, y, w, h = min(x, x1), min(y, y1), max(x+w, x1+w1) - x, max(y+h, y1+h1) - y

    # Apply tolerance
    x = max(x - tolerance, 0)
    y = max(y - tolerance, 0)
    w = min(w + 2 * tolerance, saliency_map.shape[-1] - x)
    h = min(h + 2 * tolerance, saliency_map.shape[-2] - y)

    return x, y, w, h

def find_deeplab_bounding_box(deeplab_output):
    """
    Find the bounding box around all features found by DeepLabV3.
    """
    # Convert the output to a numpy array
    output_np = deeplab_output.byte().cpu().numpy()

    # Find contours and the bounding box
    contours, _ = cv2.findContours(output_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, 0, 0  # No contours found

    x, y, w, h = cv2.boundingRect(contours[0])
    for cnt in contours[1:]:
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        x, y, w, h = min(x, x1), min(y, y1), max(x+w, x1+w1) - x, max(y+h, y1+h1) - y

    return x, y, w, h

def consolidate_bounding_boxes(boxes):
    """
    Consolidate multiple bounding boxes into a single bounding box that encompasses all.
    """
    if not boxes:
        return 0, 0, 0, 0  # Return an empty box if no boxes are provided

    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[0] + box[2] for box in boxes)
    y_max = max(box[1] + box[3] for box in boxes)

    return x_min, y_min, x_max - x_min, y_max - y_min


# Additional utility functions as needed