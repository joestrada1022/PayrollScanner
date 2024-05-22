import numpy as np
import pytesseract, cv2, argparse
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, ttk


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
    if cv2.contourArea(cnt) > min_area and cnt[:, :, 1].min() > min_area
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
timesheet = []
for box in grouped_boxes:
    # cv2.rectangle(cropped, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
    row = cropped[
        box[1] - threshold : box[3] + threshold, box[0] - threshold : box[2] + threshold
    ]
    h, w = row.shape[:2]
    left = row[:, : w // 2]
    right = row[:, w // 2 :]

    img = Image.fromarray(left)
    img2 = Image.fromarray(right)
    numConf = " -l eng --oem 3 --psm 6 digits"
    letterConf = (
        " -l eng --oem 3 --psm 3 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz"
    )
    name = pytesseract.image_to_string(img, config=letterConf)
    name = name.replace("\n", "")
    hours = pytesseract.image_to_string(img2, config=numConf)
    hours = hours.replace("\n", "")
    timesheet.insert(0, (name, hours))


class OCRCorrectionApp:
    def __init__(self, ocr_results, image):
        self.ocr_results = ocr_results
        self.image = image

        self.root = tk.Tk()
        self.root.title("OCR Correction")

        self.tree = ttk.Treeview(self.root, columns=("Name", "Hours"), show="headings")
        self.tree.heading("Name", text="Name")
        self.tree.heading("Hours", text="Hours")
        self.tree.bind("<Double-1>", self.on_double_click)
        self.tree.bind("<Return>", self.on_double_click)
        for name, hours in ocr_results:
            self.tree.insert("", "end", values=(name, hours))
        self.tree.pack(side=tk.LEFT, expand=True, fill="both")

        self.load_image()

    def on_double_click(self, event):
        item = self.tree.selection()[0]
        column = self.tree.identify_column(event.x)
        col_index = int(str(column).split("#")[-1]) - 1
        self.edit_cell(item, col_index)

    def edit_cell(self, item, col_index):
        edit_window = tk.Toplevel()
        edit_window.title("Edit Cell")

        label = tk.Label(edit_window, text="New Value:")
        label.grid(row=0, column=0, padx=10, pady=5)

        current_value = self.tree.item(item, "values")[col_index]  # Get current value
        new_value = tk.Entry(edit_window)
        new_value.insert(tk.END, current_value)  # Insert current value into entry
        new_value.grid(row=0, column=1, padx=10, pady=5)
        new_value.focus_set()  # Set focus to the entry field

        new_value.bind(
            "<Return>",
            lambda enter: self.save_edit(edit_window, item, col_index, new_value),
        )

        save_button = tk.Button(
            edit_window,
            text="Save",
            command=lambda: self.save_edit(edit_window, item, col_index, new_value),
        )
        save_button.grid(row=1, column=0, columnspan=2, padx=10, pady=5)

    def save_edit(self, edit_window, item, col_index, new_value_entry):
        new_val = new_value_entry.get().strip()
        if new_val:
            self.tree.item(
                item,
                values=(
                    new_val if col_index == 0 else self.tree.item(item, "values")[0],
                    new_val if col_index == 1 else self.tree.item(item, "values")[1],
                ),
            )
            # Update the original list of tuples
            index = int(self.tree.index(item))
            self.ocr_results[index] = (
                new_val if col_index == 0 else self.tree.item(item, "values")[0],
                new_val if col_index == 1 else self.tree.item(item, "values")[1],
            )
            edit_window.destroy()
        else:
            messagebox.showerror("Error", "Value cannot be empty!")

    def load_image(self):
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image_pil)
        label = tk.Label(self.root, image=photo)
        label.image = photo  # Keep a reference to the image to prevent it from being garbage collected
        label.pack(side=tk.RIGHT, expand=True, fill="both")

    def run(self):
        self.root.mainloop()


app = OCRCorrectionApp(timesheet, cropped)
app.run()
print(timesheet)
