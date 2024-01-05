from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)

# def decode_predictions(scores, geometry):
#     # grab the number of rows and columns from the scores volume, then
#     # initialize our set of bounding box rectangles and corresponding
#     # confidence scores
#     (numRows, numCols) = scores.shape[2:4]
#     rects = []
#     confidences = []

#     # loop over the number of rows
#     for y in range(0, numRows):
#         # extracts the scores, then the data for the bounding boxes
#         scoresData = scores[0, 0, y]
#         xData0 = geometry[0, 0, y]
#         xData1 = geometry[0, 1, y]
#         xData2 = geometry[0, 2, y]
#         xData3 = geometry[0, 3, y]
#         anglesData = geometry[0, 4, y]

#         # loop over the number of columns
#         for x in range(0, numCols):
#             # ignore low scores
#             if scoresData[x] < args["min_confidence"]:
#                 continue

#             (offsetX, offsetY) = (x * 4.0, y * 4.0)

#             # extract the rotation angle for the prediction then computer sin and cos
#             angle = anglesData[x]
#             cos = np.cos(angle)
#             sin = np.sin(angle)

#             # use the geometry volume to derive the width and height
#             h = xData0[x] + xData2[x]
#             w = xData1[x] + xData3[x]

#             # compute both the starting and ending (x, y) coords for the bounding box
#             endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
#             endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
#             startX = int(endX - w)
#             startY = int(endY - h)

#             # add the bounding boxes coords and their score to the lists
#             rects.append((startX, startY, endX, endY))
#             confidences.append(scoresData[x])

#     return (rects, confidences)


# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str,
# 	help="path to input image")
# ap.add_argument("-east", "--east", type=str,
# 	help="path to input EAST text detector")
# ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
# 	help="minimum probability required to inspect a region")
# ap.add_argument("-w", "--width", type=int, default=320,
# 	help="nearest multiple of 32 for resized width")
# ap.add_argument("-e", "--height", type=int, default=320,
# 	help="nearest multiple of 32 for resized height")
# ap.add_argument("-p", "--padding", type=float, default=0.0,
# 	help="amount of padding to add to each border of ROI")
# args = vars(ap.parse_args())

# # load the input image and grab the image dimensions
# image = cv2.imread(args["image"])
# orig = image.copy()
# (origH, origW) = image.shape[:2]

# # set new width and height and determine change ratio
# (newW, newH) = (args["width"], args["height"])
# rW = origW / float(newW)
# rH = origH / float(newH)

# # resize the image and grab new dimensions
# image = cv2.resize(image, (newW, newH))
# (H, W) = image.shape[:2]

# # define layer names for the EAST detector model
# # first is output score, second is bounding box coords
# layerNames = [
#     "feature_fusion/Conv_7/Sigmoid",
#     "feature_fusion/concat_3"]
# # load the pre-trained EAST text detector
# print("loading EAST test detector...")
# net = cv2.dnn.readNet(args["east"])

# blob = cv2.dnn.blobFromImage(image, 1.0, (W,H),
#     (123.68, 116.78, 103.94), swapRB=True, crop=False)
# net.setInput(blob)

# # define the two output layer names for the EAST detector model that
# # we are interested in -- the first is the output probabilities and the
# # second can be used to derive the bounding box coordinates of text
# (scores, geometry) = net.forward(layerNames)
# # decode the predictions, then apply NMS (non-maxima suppression) to suppress overlapping boxes
# (rects, confidences) = decode_predictions(scores, geometry)
# boxes = non_max_suppression(np.array(rects), probs=confidences)

# # initializes the list of results
# results = []

# # loop over the bounding boxes
# for(startX, startY, endX, endY) in boxes:
#     startX = int(startX * rW)
#     startY = int(startY * rH)
#     endX = int(endX * rW)
#     endY = int(endY * rH)

#     # apply padding to get a better OCR result
#     dX = int((endX - startX) * args["padding"])
#     dY = int((endY - startY) * args["padding"])

#     # apply the padding to the bounding box
#     startX = max(0, startX - dX)
#     startY = max(0, startY - dY)
#     endX = min(origW, endX + (dX * 2))
#     endY = min(origH, endY + (dY * 2))

#     # extract the ROI (region of interest)
#     roi = orig[startY:endY, startX:endX]


#     # apply Tesseract v4
#     config = ("-l eng --oem 1 --psm 7")
#     text = pytesseract.image_to_string(roi, config=config)

#     results.append(((startX, startY, endX, endY), text))

# results = sorted(results, key=lambda r:r[0][1])
# for ((startX, startY, endX, endY), text) in results:
#     # display the OCR'd text by Tesseract
#     print("OCR Text")
#     print("========")
#     print("()\n".format(text))
#     print(text)

#     # strip out non-ASCII text so we can draw the text on the image
#     # using OpenCV, then draw the text and a bounding box surrounding
#     # the text region of the input image
#     text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
#     output = orig.copy()
#     cv2.rectangle(output, (startX, startY), (endX, endY),
#         (0, 0, 255), 2)
#     cv2.putText(output, text, (startX, startY - 20),
#         cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
#     # show the output image
#     cv2.imshow("Text Detection", output)
#     cv2.waitKey(0)
image = cv2.imread("croppedHours.jpg", cv2.IMREAD_REDUCED_GRAYSCALE_4)
thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Remove horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
detect_horizontal = cv2.morphologyEx(
    thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
)
cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(thresh, cnts, [0, 0, 0])

# Remove vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
detect_vertical = cv2.morphologyEx(
    thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2
)
cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(thresh, cnts, [0, 0, 0])

result = 255 - thresh

cv2.imshow("thresh", thresh)
cv2.imshow("result", result)
cv2.waitKey()

# TODO: fix preprocessing so that image is divided into two: names and hours. the ocr each one individually
img = Image.fromarray(image)
config = " -l eng --oem 3 --psm 4"
text = pytesseract.image_to_string(img, config=config)
with open("output2.txt", "w") as f:
    f.write(text)
