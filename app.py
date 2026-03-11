import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import os
import threading
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------
# CONFIG
# -------------------
IMG_SIZE = 224
MODEL_PATH = "asl_hg_mobilenet.h5"
TEST_DATASET = "test"

# -------------------
# LOAD MODEL
# -------------------
model = tf.keras.models.load_model(MODEL_PATH)
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
model.predict(dummy)

# -------------------
# CLASSES
# -------------------
class_names = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
]

# -------------------
# PREPROCESS FUNCTION
# -------------------
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

# -------------------
# TKINTER ROOT
# -------------------
root = tk.Tk()
root.title("ASL Recognition System")
root.geometry("1100x750")
root.configure(bg="#f4f6f8")

# -------------------
# HEADER
# -------------------
header = tk.Frame(root, bg="#ffffff", bd=0)
header.pack(fill="x")
tk.Label(header, text="ASL Recognition System", font=("Segoe UI", 28, "bold"),
         bg="#ffffff", fg="#2c7be5").pack(pady=15)

tk.Label(header, text="Upload Image or Use Webcam", font=("Segoe UI", 14),
         bg="#ffffff", fg="gray").pack(pady=5)

# -------------------
# MAIN FRAME
# -------------------
main = tk.Frame(root, bg="#f4f6f8")
main.pack(fill="both", expand=True, padx=20, pady=20)

# Image Card
image_card = tk.Frame(main, bg="white", width=400, height=400,
                      highlightbackground="#ddd", highlightthickness=1)
image_card.grid(row=0, column=0, padx=30, sticky="nsew")
image_card.pack_propagate(False)
image_label = tk.Label(image_card, text="Camera / Image Preview",
                       bg="white", font=("Segoe UI", 14), fg="gray")
image_label.pack(expand=True)

# Result Card
result_card = tk.Frame(main, bg="white", width=400, height=400,
                       highlightbackground="#ddd", highlightthickness=1)
result_card.grid(row=0, column=1, padx=30, sticky="nsew")
result_card.pack_propagate(False)
tk.Label(result_card, text="Prediction", font=("Segoe UI", 20, "bold"), bg="white").pack(pady=20)
prediction = tk.Label(result_card, text="-", font=("Segoe UI", 70, "bold"),
                      fg="#2c7be5", bg="white")
prediction.pack()
confidence = tk.Label(result_card, text="Confidence: -", font=("Segoe UI", 14), bg="white")
confidence.pack(pady=10)

# Make grid expandable
main.columnconfigure(0, weight=1)
main.columnconfigure(1, weight=1)
main.rowconfigure(0, weight=1)

# -------------------
# CAMERA VARIABLES
# -------------------
camera_on = False
cap = None

# -------------------
# FUNCTIONS
# -------------------
def upload_image():
    global camera_on
    camera_on = False
    path = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.png *.jpeg")])
    if path == "":
        return
    img = Image.open(path).convert("RGB")
    display = img.resize((350,350))
    imgTk = ImageTk.PhotoImage(display)
    image_label.config(image=imgTk, text="")
    image_label.image = imgTk

    img_np = np.array(img)
    img_input = preprocess(img_np)[np.newaxis,...]
    pred = model.predict(img_input, verbose=0)
    cls = np.argmax(pred)
    conf = np.max(pred)*100
    letter = class_names[cls]

    prediction.config(text=letter)
    confidence.config(text=f"Confidence: {conf:.1f}%")

def start_camera():
    global camera_on, cap
    camera_on = True
    cap = cv2.VideoCapture(0)
    update_camera()

def update_camera():
    global camera_on
    if camera_on:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame,1)
            h,w,_ = frame.shape
            x1=int(w*0.55); y1=int(h*0.2); x2=int(w*0.9); y2=int(h*0.7)
            roi = frame[y1:y2,x1:x2]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            if roi.size!=0:
                img_input = preprocess(roi)[np.newaxis,...]
                pred = model.predict(img_input, verbose=0)
                cls = np.argmax(pred)
                conf = np.max(pred)*100
                letter = class_names[cls]
                prediction.config(text=letter)
                confidence.config(text=f"Confidence: {conf:.1f}%")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).resize((350,350))
            imgtk = ImageTk.PhotoImage(img)
            image_label.imgtk = imgtk
            image_label.configure(image=imgtk, text="")
        root.after(10, update_camera)

def stop_camera():
    global camera_on, cap
    camera_on = False
    if cap: cap.release()
    image_label.config(image="", text="Camera / Image Preview")

def clear_all():
    stop_camera()
    prediction.config(text="-")
    confidence.config(text="Confidence: -")

# -------------------
# MODEL METRICS
# -------------------
def show_metrics():
    loading = tk.Toplevel(root)
    loading.title("Processing")
    tk.Label(loading, text="Calculating model metrics...\nPlease wait",
             font=("Segoe UI", 14)).pack(padx=30, pady=30)
    loading.update()

    X_test, y_true = [], []

    for label in class_names:
        folder = os.path.join(TEST_DATASET, label)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder)[:20]:
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            X_test.append(preprocess(img))
            y_true.append(label)

    if len(X_test) == 0:
        loading.destroy()
        messagebox.showerror("Error", "No test images found!")
        return

    X_test = np.array(X_test)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = [class_names[np.argmax(p)] for p in y_pred_probs]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    loading.destroy()

    metrics_window = tk.Toplevel(root)
    metrics_window.title("Model Performance Dashboard")
    metrics_window.geometry("1200x900")

    canvas = tk.Canvas(metrics_window)
    scrollbar = tk.Scrollbar(metrics_window, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0,0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    tk.Label(scroll_frame, text="Model Performance Dashboard",
             font=("Segoe UI", 20, "bold")).pack(pady=10)
    tk.Label(scroll_frame, text=f"Model Accuracy: {acc*100:.2f}%",
             font=("Segoe UI", 16, "bold"), fg="green").pack(pady=5)

    # Confusion matrix
    fig1 = plt.Figure(figsize=(12,8))
    ax1 = fig1.add_subplot(111)
    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                cbar=False, ax=ax1)
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    canvas1 = FigureCanvasTkAgg(fig1, scroll_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(pady=10)

    # Pie chart
    fig2 = plt.Figure(figsize=(8,6))
    ax2 = fig2.add_subplot(111)
    pred_counts = pd.Series(y_pred).value_counts()
    ax2.pie(pred_counts, labels=pred_counts.index, autopct="%1.1f%%")
    ax2.set_title("Prediction Distribution")
    canvas2 = FigureCanvasTkAgg(fig2, scroll_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack(pady=10)

    # Class-wise accuracy
    fig3 = plt.Figure(figsize=(12,6))
    ax3 = fig3.add_subplot(111)
    class_acc = []
    for label in class_names:
        idx = [i for i, x in enumerate(y_true) if x == label]
        if len(idx) == 0:
            class_acc.append(0)
        else:
            correct = sum([1 for i in idx if y_pred[i] == label])
            class_acc.append(correct / len(idx))
    ax3.bar(class_names, class_acc)
    ax3.set_title("Class Accuracy")
    ax3.set_xlabel("Class")
    ax3.set_ylabel("Accuracy")
    ax3.tick_params(axis='x', rotation=90)
    canvas3 = FigureCanvasTkAgg(fig3, scroll_frame)
    canvas3.draw()
    canvas3.get_tk_widget().pack(pady=10)

    tk.Label(scroll_frame, text="Classification Report", font=("Segoe UI", 16, "bold")).pack(pady=5)
    report_box = tk.Text(scroll_frame, height=15, width=120)
    report_box.pack(pady=10)
    report_box.insert(tk.END, report)

# -------------------
# BUTTONS FRAME
# -------------------
buttons = tk.Frame(root, bg="#f4f6f8")
buttons.pack(pady=20, fill="x")

button_specs = [
    ("Upload Image", "#2c7be5", upload_image),
    ("Start Camera", "#28a745", start_camera),
    ("Stop Camera", "#ff9800", stop_camera),
    ("Clear", "#e63757", clear_all),
    ("Model Metrics", "#6f42c1", lambda: threading.Thread(target=show_metrics).start())
]

for i, (text, color, func) in enumerate(button_specs):
    btn = tk.Button(buttons, text=text, font=("Segoe UI", 15, "bold"),
                    bg=color, fg="white", bd=0, padx=20, pady=10, command=func)
    btn.pack(side="left", padx=10, expand=True, fill="x")

# -------------------
# FOOTER
# -------------------
tk.Label(root, text="Deep Learning ASL Recognition System",
         bg="#f4f6f8", fg="gray").pack(side="bottom", pady=10)

root.mainloop()