import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt

# ==== SETTINGS ====
CLASSES = ['cat', 'apple', 'cup', 'axe', 'book']
MODEL_PATH = 'doodle_model.h5'

# ==== LOAD MODEL ====
if os.path.exists(MODEL_PATH):q
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print(f"[WARNING] Model file '{MODEL_PATH}' not found. Predictions won't work.")
    model = None

# ==== DRAWING CONFIG ====
canvas = np.zeros((280, 280, 3), dtype=np.uint8)
drawing = False
brush_size = 10
eraser_mode = False
live_mode = False
color = (255, 255, 255)
undo_stack = []

# ==== DRAW FUNCTION ====
def draw(event, x, y, flags, param):
    global drawing, canvas, undo_stack

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        undo_stack.append(canvas.copy())  # Save state
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), brush_size, (0, 0, 0) if eraser_mode else color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right-click to erase
        cv2.circle(canvas, (x, y), brush_size, (0, 0, 0), -1)

# ==== WINDOW SETUP ====
cv2.namedWindow("AI Doodle Classifier")
cv2.setMouseCallback("AI Doodle Classifier", draw)

# ==== PREDICTION ====
def predict_doodle(img):
    if model is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (28, 28)).astype(np.float32) / 255.0
    img_input = img_resized.reshape(1, 28, 28, 1)
    pred = model.predict(img_input)[0]
    top_indices = pred.argsort()[-3:][::-1]
    results = [(CLASSES[i], pred[i]) for i in top_indices]
    return results

# ==== DISPLAY RESULTS ====
def show_prediction(results):
    bar = np.zeros((150, 280, 3), dtype=np.uint8)
    for i, (label, conf) in enumerate(results):
        text = f"{label}: {conf:.2f}"
        cv2.putText(bar, text, (10, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(bar, (150, 25 + i * 40), (150 + int(conf * 120), 45 + i * 40), (0, 255, 0), -1)
    return bar

# ==== PLOT GRAPH ====
def show_prediction_graph(results):
    labels = [label for label, _ in results]
    values = [conf for _, conf in results]

    plt.barh(labels, values, color='skyblue')
    plt.xlabel("Confidence")
    plt.title("Doodle Predictions")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

# ==== HELP ====
instructions = """
INSTRUCTIONS:
 - Draw with left mouse
 - Right click = Erase
 - Press 'e' to toggle Eraser mode
 - '+'/'-' to adjust brush size
 - 'p' to predict
 - 'g' to show graph
 - 'l' to toggle live prediction
 - 'u' to undo
 - 's' to save with timestamp
 - 'r' to draw a random test image
 - 'c' to clear
 - 'q' to quit
"""
print(instructions)

# ==== MAIN LOOP ====
while True:
    results = predict_doodle(canvas) if live_mode else None
    canvas_display = canvas.copy()
    if results:
        pred_bar = show_prediction(results)
        pred_bar_resized = cv2.resize(pred_bar, (canvas_display.shape[1], pred_bar.shape[0]))
        combined = np.vstack((canvas_display, pred_bar_resized))
    else:
        combined = canvas_display

    cv2.imshow("AI Doodle Classifier", combined)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[:] = 0
    elif key == ord('e'):
        eraser_mode = not eraser_mode
        print(f"Eraser: {'ON' if eraser_mode else 'OFF'}")
    elif key == ord('+') and brush_size < 50:
        brush_size += 2
        print(f"Brush size: {brush_size}")
    elif key == ord('-') and brush_size > 2:
        brush_size -= 2
        print(f"Brush size: {brush_size}")
    elif key == ord('p'):
        results = predict_doodle(canvas)
        print("Top Predictions:")
        for label, conf in results:
            print(f" - {label}: {conf:.2f}")
    elif key == ord('g'):
        results = predict_doodle(canvas)
        if results:
            show_prediction_graph(results)
    elif key == ord('l'):
        live_mode = not live_mode
        print(f"Live Mode: {'ON' if live_mode else 'OFF'}")
    elif key == ord('s'):
        filename = f"doodle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, canvas)
        print(f"Doodle saved as {filename}")
    elif key == ord('u') and undo_stack:
        canvas = undo_stack.pop()
        print("Undid last stroke.")
    elif key == ord('r'):
        canvas[:] = np.random.randint(0, 256, canvas.shape, dtype=np.uint8)
        print("Random image for testing applied!")

cv2.destroyAllWindows()