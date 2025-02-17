## **_Image_Detection Using YOLO Model_**
This project leverages the YOLOv5 model to perform object detection on images. It provides a GUI interface where users can upload an image, detect objects in the image, and download the processed image with bounding boxes. The detected object details are saved in a CSV file for further analysis.

## Libraries Used
- **torch**: A deep learning framework used for training and inference. It provides easy-to-use tools for creating machine learning models, including pre-trained models like YOLOv5.
- **Pandas**: A data manipulation and analysis library that allows you to read, write, and manipulate structured data in Python.
- **Pillow (PIL)**: A Python Imaging Library used to open, manipulate, and save image files.
- **Tkinter**: A GUI library in Python for creating interactive applications with widgets like buttons, labels, and text fields.
- **shutil**: A module to perform high-level file operations like copying or moving files.
- **os**: A library used for interacting with the operating system, such as managing file paths and directories.

## Features

1. **Image Upload**: Select an image file to perform object detection.
2. **Bounding Box Drawing**: Draws bounding boxes around detected objects.
3. **CSV Output**: Saves the detection details (object names, confidence, and coordinates) into a CSV file.
4. **Download Processed Image**: Provides an option to download the image with bounding boxes.
5. **Duplicate Image Handling**: Prevents the detection process from being repeated if an image with bounding boxes already exists.
## Steps to Run the Project

### 1. Clone the Repository
```bash
git clone [GitHub](https://github.com/anantbhadani/Image_Detection-Using-YOLO-Model.git)
cd Image_Detection
```
### 2. Create a Virtual Environment
To set up a virtual environment and avoid dependency conflicts:

```bash
python -m venv venv
```
### 3. Activate the Virtual Environment
Windows:
```bash
.\venv\Scripts\activate
```
Mac/Linux:
```bash
source venv/bin/activate
```
### 4. Install Required Libraries
Once the virtual environment is activated, install the necessary libraries:
```bash
pip install -r requirements.txt
```
Note: Create a requirements.txt file with the following content:
```python
torch
torchvision
pandas
matplotlib
ultralytics
Pillow
tensorflow
```
### 5. Run the Application
After installing the required libraries, run the Python script:

```bash
python app.py
```
This will launch a window with buttons to upload images, detect objects, and download the processed image with bounding boxes.

### Example Code Snippets
### Initialize YOLOv5 Model
```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  
# YOLOv5s pre-trained model
```
This code loads the pre-trained YOLOv5 model (yolov5s), which is a lightweight version of the YOLOv5 object detection model.

### Processing Image and Drawing Bounding Boxes
```python
results = model(image)
detections = results.pandas().xyxy[0]  # Get results as a DataFrame

# Manually save image with bounding boxes
img_with_boxes = Image.open(image_path)
draw = ImageDraw.Draw(img_with_boxes)  # Initialize ImageDraw to draw on the image
for _, row in detections.iterrows():
    draw.rectangle([row['xmin'], row['ymin'], row['xmax'], row['ymax']], outline="red", width=3)
img_with_boxes.save(bounding_box_image_path)
```
This code detects objects in the image and draws red bounding boxes around them. It saves the image with bounding boxes and provides the option to download it.

### Handling Duplicate Images
```python
bounding_box_image_path = os.path.join(BOUNDING_BOXES_DIR, image_name)

if os.path.isfile(bounding_box_image_path):
    messagebox.showinfo("Info", f"Image with bounding boxes already exists: {bounding_box_image_path}")
    # Load and display the existing image with bounding boxes
    img_with_boxes = Image.open(bounding_box_image_path)
    img_with_boxes.thumbnail((400, 400))  # Resize to fit the screen
    img_with_boxes_tk = ImageTk.PhotoImage(img_with_boxes)

    # Update Label widget with the image
    image_label.config(image=img_with_boxes_tk)
    image_label.image = img_with_boxes_tk
```
This block of code ensures that if an image with bounding boxes already exists, the program won't duplicate the detection process, and it will simply display the existing image with bounding boxes.

### Troubleshooting
Issue with Image Not Showing Up: Make sure that the image path is correct and that the file exists. Check the directory paths used for saving images and CSV files.

**Dependencies:** Ensure all libraries are installed in your virtual environment using the ```pip install -r requirements.txt command.```#   I m a g e _ D e t e c t i o n - U s i n g - Y O L O - M o d e l  
 #   I m a g e _ D e t e c t i o n - U s i n g - Y O L O - M o d e l  
 #   I m a g e _ D e t e c t i o n - U s i n g - Y O L O - M o d e l  
 