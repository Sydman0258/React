import cv2
import numpy as np

def process_image(image_bytes, filter_type, width=None, height=None, brightness=None):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Resize
    if width and height:
        img = cv2.resize(img, (width, height))

    # Brightness
    if brightness is not None:
        img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)

    # Filters
    if filter_type == "grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif filter_type == "blur":
        img = cv2.GaussianBlur(img, (15, 15), 0)

    elif filter_type == "rotate":
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()