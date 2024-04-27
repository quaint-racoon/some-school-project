import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np

click = False
angles = []
startedaverage = True
a = None
b = None
text = ""
sampleimg = None
frameimg = None
photo_images = []  # List to store references to PhotoImage objects
def rescaleFrame(frame, canvas_width=800, canvas_height=600):
    # Get the dimensions of the frame
    height, width = frame.shape[:2]

    # Calculate the scaling factors for width and height
    width_scale = canvas_width / width
    height_scale = canvas_height / height

    # Choose the smaller scaling factor to fit the entire image within the canvas
    scale = min(width_scale, height_scale)

    # Resize the frame with the calculated scale
    new_width = int(width * scale)
    new_height = int(height * scale)
    dimensions = (new_width, new_height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def findangle(a, b):
    if (b[0] - a[0]) == 0:
        return
    c = abs((np.arctan((b[1] - a[1]) / (b[0] - a[0])) * (180.0 / np.pi)))
    angles.append(c)
    if not startedaverage:
        return "angle : " + str(round(c, 2)) + " || not calculating average"
    d = findavg()
    return "angle: "+str(round(c, 2)) + " || angle case: " + findscoliosis(c)


def findscoliosis(a):
    if 0 <= a <= 1.5:
        return "normal"
    if 1.5 < a <= 4:
        return "possible moderate scoliosis"
    if 4 < a :
        return "possible severe scoliosis"

def findavg():
    return np.mean(angles)

def mousePoints(event, sampleimg, param=None):
    global click, a, b
    x, y = event.x, event.y  # Extract mouse coordinates from the event
    if event.type == tk.EventType.ButtonPress:  # Check for left button down event
        if not click:
            sampleimg = sampleimg_original.copy()
            update_image(sampleimg)
            if sampleimg is not None:  # Check if sampleimg is initialized
                # Adjust coordinates based on resizing factor
                x_resized = int(x * (sampleimg.shape[1] / canvas.winfo_width()))
                # Adjust y-coordinate based on the difference between canvas height and image height
                y_resized = int(y * (sampleimg.shape[0] / canvas.winfo_height()))
                y_resized = min(y_resized, sampleimg.shape[0])  # Ensure y_resized does not exceed image height
                a = (x_resized, y_resized)
                click = True
                cv.circle(sampleimg, a, 5, (0, 0, 255), cv.FILLED) 
                update_image(sampleimg)
        else:
            if sampleimg is not None:  # Check if sampleimg is initialized
                # Adjust coordinates based on resizing factor
                x_resized = int(x * (sampleimg.shape[1] / canvas.winfo_width()))
                # Adjust y-coordinate based on the difference between canvas height and image height
                y_resized = int(y * (sampleimg.shape[0] / canvas.winfo_height()))
                y_resized = min(y_resized, sampleimg.shape[0])  # Ensure y_resized does not exceed image height
                b = (x_resized, y_resized)
                click = False
                  # Restore the original image
                # Draw red circle at point b
                cv.circle(sampleimg, b, 5, (0, 0, 255), cv.FILLED)
                angle_info = findangle(a, b)
                label.config(text=angle_info)
                update_image(sampleimg)
                # Redraw original image after two clicks
                a = None
                b = None


def update_image(sampleimg):
    global frameimg, photo_images, canvas

    if sampleimg is None:
        print("Error: Empty sample image.")
        return

    try:
        
        # Resize the image to fit the canvas
        frameimg = rescaleFrame(sampleimg, canvas.winfo_width(), canvas.winfo_height())

        # Create Tkinter image from the resized frame
        img_tk = cv2_to_tkinter(frameimg)

        # Update the canvas size to match the resized image
        canvas.config(width=sampleimg_original.shape[1], height=sampleimg_original.shape[0])

        # Clear the canvas and draw the resized image
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        # Store the Tkinter image reference
        canvas.image = img_tk
        photo_images.append(img_tk)

        # Bind mouse callback after displaying the image
        canvas.bind("<Button-1>", lambda event: mousePoints(event, sampleimg))

    except Exception as e:
        print("Error during image update:", e)



def mouse_click(event):
    global click, a, b
    if click:
        click = False
        b = (event.x, event.y)
        canvas.create_oval(b[0] - 5, b[1] - 5, b[0] + 5, b[1] + 5, outline="red", width=2)
        angle_info = findangle(a, b)
        label.config(text=angle_info)
    else:
        click = True
        a = (event.x, event.y)
        canvas.create_oval(a[0] - 5, a[1] - 5, a[0] + 5, a[1] + 5, outline="red", width=2)




# Modify the load_image function to set the mouse callback
def load_image():
    global sampleimg, sampleimg_original
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.svg;*.jfif")])
    if file_path:
        image = cv.imread(file_path)
        image = rescaleFrame(image, 700)
        sampleimg = image.copy()
        sampleimg_original = sampleimg.copy()  # Store a copy of the original image
        update_image(sampleimg)


def live_camera():
    cap = cv.VideoCapture(0)
    paused = False

    def toggle_pause(event):
        nonlocal paused
        paused = not paused
       
    root.bind("<space>", toggle_pause)

    def update_frame():
        nonlocal paused
        if not paused:
            ret, frame = cap.read()
            global sampleimg_original
            sampleimg_original=frame
            if ret:
                sampleimg = rescaleFrame(frame, 1000)
                update_image(sampleimg)
        root.after(10, update_frame)  # Update frame every 10 milliseconds

    update_frame()

    root.mainloop()  # Start the tkinter event loop

    cap.release()



def cv2_to_tkinter(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    img_tk = ImageTk.PhotoImage(image=image)
    return img_tk

def close_app(event):
    global done
    done = True
    root.destroy()
# GUI

root = tk.Tk()

root.configure(background="#d8add0")
root.title("Angle Measurement")
root.bind("<Key-q>", close_app)

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

canvas = tk.Canvas(root, width=800, height=600,background="#e8dfee")
canvas.pack()
canvas.bind("<Button-1>", mouse_click)

label = tk.Label(root, text="",background="#d8add0",font=("",15))
label.pack()

btn_image = tk.Button(frame, text="Select Image", command=load_image,background="#fcf9ce")
btn_image.pack(side=tk.LEFT, padx=5)

btn_camera = tk.Button(frame, text="Open Live Camera", command=live_camera,background="#fcf9ce")
btn_camera.pack(side=tk.LEFT, padx=5)

root.mainloop()
