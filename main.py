from scanner import Scanner
from validator import OCRValidator

import pytesseract, argparse, cv2
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = (
    "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)


def csv_parser(timesheet, template_path, output_path):
    df = pd.read_csv(template_path)
    for first_name, hours in timesheet:
        row_index = df.loc[df["first_name"].str.lower() == first_name.lower()]

        # check if row_index is empty
        if row_index.empty:
            print(f"{first_name} not found in the dataframe")
            continue
        else:
            row_index = row_index.index[0]
        hours_index = df.columns.get_loc("regular_hours")

        df.iat[row_index, hours_index] = float(hours)

        reimbursement_index = df.columns.get_loc("reimbursement")
        df.iat[row_index, reimbursement_index] = 2 * float(hours)

    # overwrite df to csv
    df.to_csv(output_path, index=False)
    print("\n\nSuccess!")


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

    return timesheet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="automating hours entry through OCR")
    parser.add_argument(
        "ImagePath", help="Path to the image file", metavar="path", type=str
    )
    parser.add_argument(
        "--template", help="Path to the payroll template file", metavar="path", type=str
    )
    parser.add_argument("--output", help="output destination", metavar="path", type=str)
    args = parser.parse_args()

    if args.ImagePath is None:
        print("Usage: python scanner.py <image>")
        exit()
    else:
        scanned = cv2.imread(args.ImagePath, cv2.IMREAD_REDUCED_COLOR_4)

    timesheet = timesheet_adjuster(scanned)
    csv_parser(timesheet, args.template, args.output)


# TODO: add list for people who don't get reimbursed
# TODO: before validator, autocorrect names based on distance to names given on template
# TODO: make code more readable (comments, spacing, etc)
