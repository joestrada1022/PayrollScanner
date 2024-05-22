from scanner import Scanner
from validator import OCRValidator
import pytesseract, argparse, cv2

pytesseract.pytesseract.tesseract_cmd = (
    "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)


def timesheet_adjuster(scanned_image):
    scanner = Scanner()
    mask, erosion = scanner.process(scanned_image)
    cropped, boxes = scanner.get_bbox(mask, erosion)
    timesheet = scanner.generate_timesheet(boxes, cropped)
    print(timesheet)
    print("-" * 40)
    validator = OCRValidator(timesheet, cropped)
    validator.run()
    print(timesheet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="automating hours entry through OCR")
    parser.add_argument(
        "ImagePath", help="Path to the image file", metavar="path", type=str
    )
    args = parser.parse_args()

    if args.ImagePath is None:
        print("Usage: python scanner.py <image>")
        exit()
    else:
        scanned = cv2.imread(args.ImagePath, cv2.IMREAD_REDUCED_COLOR_4)

    timesheet_adjuster(scanned)
