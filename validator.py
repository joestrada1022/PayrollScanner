from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import messagebox, ttk


class OCRValidator:
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
