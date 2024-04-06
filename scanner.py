import numpy as np
import pytesseract, cv2, argparse
from PIL import Image
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="automating hours entry through OCR")
parser.add_argument(
    "ImagePath", help="Path to the image file", metavar="path", type=str
)
args = parser.parse_args()

if args.ImagePath is None:
    print("Usage: python scanner.py <image>")
    exit()


pytesseract.pytesseract.tesseract_cmd = (
    "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)


def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lowerLimit = hsvC[0][0][0] - 10, 100, 100
    upperLimit = hsvC[0][0][0] + 10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit


lower, upper = get_limits(color=(245, 0, 0))
image = cv2.imread(args.ImagePath, cv2.IMREAD_REDUCED_COLOR_4)
image = image[: int(image.shape[0] * 0.85), :]
image = cv2.bilateralFilter(image, 11, 17, 17)
# original was 127-255
thresh = cv2.threshold(image, 145, 255, cv2.THRESH_BINARY)[1]

hsvImage = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsvImage, lower, upper)

binr = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
kernel = np.ones((2, 2), np.uint8)
erosion = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imshow("image", erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

pil_img = Image.fromarray(erosion)

bbox = pil_img.getbbox()

if bbox is not None:
    x1, y1, x2, y2 = bbox
    cropped = mask[y1 + 10 : y2 + 10, x1 - 10 : x2 - 10]
    cv2.imshow("cropped", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

min_area = 10
contours, _ = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = [
    cnt
    for cnt in contours
    if cv2.contourArea(cnt) > min_area and cnt[:, :, 1].min() > 10
]
# Draw bounding boxes around remaining contours to isolate rows of text
cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
# Group bounding boxes along the vertical axis
grouped_boxes = []
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    grouped = False
    for box in grouped_boxes:
        if abs(y - box[1]) < 20:  # adjust this threshold as needed
            box[0] = min(box[0], x)
            box[1] = min(box[1], y)
            box[2] = max(box[2], x + w)
            box[3] = max(box[3], y + h)
            grouped = True
            break
    if not grouped:
        grouped_boxes.append([x, y, x + w, y + h])

threshold = 4
# Draw bounding boxes around grouped boxes to isolate rows of text
for box in grouped_boxes:
    # cv2.rectangle(cropped, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
    row = cropped[
        box[1] - threshold : box[3] + threshold, box[0] - threshold : box[2] + threshold
    ]
    h, w = row.shape[:2]
    left = row[:, : w // 2]
    right = row[:, w // 2 :]
    cv2.imshow("left", left)
    cv2.imshow("right", right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = Image.fromarray(left)
    img2 = Image.fromarray(right)
    numConf = " -l eng --oem 3 --psm 6 digits"
    letterConf = (
        " -l eng --oem 3 --psm 3 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz"
    )
    text = pytesseract.image_to_string(img, config=letterConf)
    text2 = pytesseract.image_to_string(img2, config=numConf)
    print((text, text2))
    # TODO isntead of printing, add to dictionary
