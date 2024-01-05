import numpy as np
import pytesseract
import cv2
from PIL import Image

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
image = cv2.imread("croppedHours2.jpg", cv2.IMREAD_REDUCED_COLOR_4)
# original was 127-255
thresh = cv2.threshold(image, 148, 255, cv2.THRESH_BINARY)[1]

hsvImage = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsvImage, lower, upper)
cv2.imshow("image", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# h, w = image.shape
# left = mask[:, : w // 2]
# right = image[:, w // 2 :]
# cv2.imshow("Left", left)
# cv2.imshow("Right", right)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = Image.fromarray(left)
# img2 = Image.fromarray(right)

numConf = " -l eng --oem 3 --psm 6 -c tessedit_char_whitelist=0123456789."
letterConf = " -l eng --oem 3 --psm 3 -c tessedit_char_blacklist=|-_[]I~"
# text = pytesseract.image_to_string(img, config=letterConf)
# text2 = pytesseract.image_to_string(img2, config=numConf)
text = pytesseract.image_to_string(image, config=letterConf)

with open("output.txt", "w") as f:
    f.write(text)
