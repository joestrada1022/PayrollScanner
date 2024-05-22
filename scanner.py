import cv2, pytesseract
import numpy as np
from PIL import Image


class Scanner:
    def __init__(self, min_area=10, text_color=(245, 0, 0)):
        self.min_area = min_area
        self.text_color = text_color

    def generate_timesheet(grouped_boxes, cropped, padding=4):
        timesheet = []
        for box in grouped_boxes:
            # cv2.rectangle(cropped, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            row = cropped[
                box[1] - padding : box[3] + padding,
                box[0] - padding : box[2] + padding,
            ]
            _, w = row.shape[:2]
            left = row[:, : w // 2]
            right = row[:, w // 2 :]

            img = Image.fromarray(left)
            img2 = Image.fromarray(right)
            numConf = " -l eng --oem 3 --psm 6 digits"
            letterConf = " -l eng --oem 3 --psm 3 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz"
            name = pytesseract.image_to_string(img, config=letterConf)
            name = name.replace("\n", "")
            hours = pytesseract.image_to_string(img2, config=numConf)
            hours = hours.replace("\n", "")
            timesheet.insert(0, (name, hours))

        return timesheet

    def get_bbox(self, erosion, mask):
        erosion = Image.fromarray(erosion)
        bbox = erosion.getbbox()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cropped = mask[y1 + 10 : y2 + 10, x1 - 10 : x2 - 10]

        contours, _ = cv2.findContours(
            cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filtered_contours = [
            cnt
            for cnt in contours
            if cv2.contourArea(cnt) > self.min_area
            and cnt[:, :, 1].min() > self.min_area
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

        return cropped, grouped_boxes

    def process(self, image):
        lower, upper = self._get_color_bounds(color=(245, 0, 0))
        image = image[: int(image.shape[0] * 0.85), :]
        image = cv2.bilateralFilter(image, 11, 17, 17)
        thresh = cv2.threshold(image, 145, 255, cv2.THRESH_BINARY)[1]

        hsvImage = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsvImage, lower, upper)

        binr = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = np.ones((2, 2), np.uint8)
        erosion = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask, erosion

    def _get_color_bounds(self, rgb_value):
        c = np.uint8([[rgb_value]])
        hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

        lowerLimit = hsvC[0][0][0] - 10, 100, 100
        upperLimit = hsvC[0][0][0] + 10, 255, 255

        lowerLimit = np.array(lowerLimit, dtype=np.uint8)
        upperLimit = np.array(upperLimit, dtype=np.uint8)

        return lowerLimit, upperLimit
