import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Labels (same as in your UI)
classes = ['cat', 'apple', 'cup', 'axe', 'book']

# Load training data
X, y = [], []
for idx, cls in enumerate(classes):
    data = np.load(f'{cls}.npy')[:3000]  # make sure .npy files exist
    X.append(data)
    y += [idx] * 3000

X = np.concatenate(X, axis=0)
y = np.array(y)
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)

# Define model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train and save
model.fit(X, y, epochs=5, batch_size=64)
model.save('doodle_model.h5')
print("✅ Model trained and saved as doodle_model.h5")