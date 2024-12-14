import os
import torch
import pandas as pd
from PIL import Image, ImageTk, ImageDraw  # Added ImageDraw
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import sys
import shutil

# Paths
IMAGES_DIR = "images/"
RESULTS_DIR = "results/"
CSV_DIR = os.path.join(RESULTS_DIR, "csv/")
BOUNDING_BOXES_DIR = r"results/bounding_boxes"

# Step 1: Initialize YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5s pre-trained model

# Step 2: Ensure Directories Exist
os.makedirs(CSV_DIR, exist_ok=True)  # Create results/csv if it doesn't exist
os.makedirs(BOUNDING_BOXES_DIR, exist_ok=True)  # Create results/bounding_boxes if it doesn't exist

def process_image(image_path):
    try:
        # Get the name of the image (without path)
        image_name = os.path.basename(image_path)
        bounding_box_image_path = os.path.join(BOUNDING_BOXES_DIR, image_name)

        # Check if the image with bounding boxes already exists
        if os.path.isfile(bounding_box_image_path):
            # If the image with bounding boxes exists, show it and the existing details
            messagebox.showinfo("Info", f"Image with bounding boxes already exists: {bounding_box_image_path}")

            # Load and display the saved image with bounding boxes
            img_with_boxes = Image.open(bounding_box_image_path)
            img_with_boxes.thumbnail((400, 400))  # Resize to fit the screen
            img_with_boxes_tk = ImageTk.PhotoImage(img_with_boxes)

            # Update the Label widget with the existing image
            image_label.config(image=img_with_boxes_tk)
            image_label.image = img_with_boxes_tk

            # Load and display the object detection details from the CSV file
            csv_file = os.path.join(CSV_DIR, image_name.replace(".jpg", "_details.csv"))
            if os.path.isfile(csv_file):
                detections = pd.read_csv(csv_file)
                unique_objects = detections['name'].unique()
                object_details = "\n".join([f"{i}. {obj.capitalize()}" for i, obj in enumerate(unique_objects, 1)])

                # Show detected objects in the text widget
                detection_details_text.delete(1.0, tk.END)  # Clear previous text
                detection_details_text.insert(tk.END, f"Objects Detected:\n{object_details}")
            else:
                messagebox.showwarning("Warning", f"CSV file with detection details not found for {image_name}")

            return  # Skip processing if image with bounding boxes already exists

        # If the image does not exist, process it
        image = Image.open(image_path)

        # Step 4: Perform Object Detection
        results = model(image)
        detections = results.pandas().xyxy[0]  # Get results as a DataFrame

        # Step 5: Save Detection Details to CSV
        csv_file = os.path.join(CSV_DIR, image_name.replace(".jpg", "_details.csv"))
        detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']].to_csv(csv_file, index=False)
        messagebox.showinfo("Success", f"CSV file saved at: {csv_file}")

        # Step 6: Manually Save Image with Bounding Boxes
        img_with_boxes = Image.open(image_path)
        draw = ImageDraw.Draw(img_with_boxes)  # Initialize the ImageDraw object to draw on the image
        for _, row in detections.iterrows():
            # Draw bounding box on the image
            draw.rectangle([row['xmin'], row['ymin'], row['xmax'], row['ymax']], outline="red", width=3)

        # Save the image with bounding boxes manually
        img_with_boxes.save(bounding_box_image_path)
        messagebox.showinfo("Success", f"Image with bounding boxes saved at: {bounding_box_image_path}")

        # Log the path of the saved image for verification
        print(f"Saved image with bounding boxes at: {bounding_box_image_path}")

        # Step 7: Print Detected Objects as Headings/Subheadings
        unique_objects = detections['name'].unique()
        object_details = "\n".join([f"{i}. {obj.capitalize()}" for i, obj in enumerate(unique_objects, 1)])

        # Show detected objects in the text widget
        detection_details_text.delete(1.0, tk.END)  # Clear previous text
        detection_details_text.insert(tk.END, f"Objects Detected:\n{object_details}")

        # Show the image with bounding boxes on the GUI
        img_with_boxes.thumbnail((400, 400))  # Resize to fit the screen
        img_with_boxes_tk = ImageTk.PhotoImage(img_with_boxes)

        # Update the Label widget with the new image
        image_label.config(image=img_with_boxes_tk)
        image_label.image = img_with_boxes_tk

    except Exception as e:
        # Log the error message in the terminal
        print(f"Error: {str(e)}", file=sys.stderr)
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def download_image():
    # Ask user for destination folder
    result_image_path = os.path.join(BOUNDING_BOXES_DIR, os.path.basename(image_name))

    # Ensure the image exists before trying to download
    if os.path.isfile(result_image_path):
        download_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")], initialfile=image_name)
        if download_path:
            try:
                # Copy the saved image to the selected destination
                img_with_boxes = Image.open(result_image_path)
                img_with_boxes.save(download_path)
                messagebox.showinfo("Download Success", f"Image successfully downloaded to {download_path}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while downloading the image: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No download path selected.")
    else:
        messagebox.showerror("Error", "No image found to download.")

# Step 8: GUI Setup
def upload_image():
    # Open file dialog to select an image
    global image_name
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        image_name = os.path.basename(file_path)  # Save the image name
        process_image(file_path)

# Step 9: Set up the main window for the GUI
root = tk.Tk()
root.title("Object Detection with YOLOv5")
root.geometry("600x600")
root.configure(bg="#f0f0f0")

# Create a frame for the buttons
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=20)

# Button to upload an image
upload_button = tk.Button(button_frame, text="Upload Image", command=upload_image, font=("Arial", 14), width=20)
upload_button.pack(side=tk.LEFT, padx=10)

# Button to download the image
download_button = tk.Button(button_frame, text="Download Image", command=download_image, font=("Arial", 14), width=20)
download_button.pack(side=tk.LEFT, padx=10)

# Label to display the image
image_label = tk.Label(root, bg="#f0f0f0")
image_label.pack(pady=10)

# Text widget to display object detection details
detection_details_text = tk.Text(root, height=10, width=50, font=("Arial", 12), wrap=tk.WORD)
detection_details_text.pack(pady=20)

# Start the GUI event loop
root.mainloop()
