# image_utils/bounding_box.py
import cv2


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
    x_max, y_max = x + w, y + h

    for cnt in contours[1:]:
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        x_max = max(x_max, x1 + w1)
        y_max = max(y_max, y1 + h1)
        x, y = min(x, x1), min(y, y1)

    w, h = x_max - x, y_max - y
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