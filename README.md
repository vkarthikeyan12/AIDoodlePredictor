# aiDoodle

A simple doodle recognition program powered by neural networks. This project utilizes Google's Quick, Draw! dataset to train and test models that can recognize hand-drawn sketches.

## 🧠 Project Structure

* `aiDoodle.py` – The main program to run the doodle recognition interface or prediction script.
* `train.py` – Script used to train the model using `.npy` data files from the Quick, Draw! dataset.
* `.idea/` – Project configuration files for JetBrains PyCharm.
* `.gitignore` – Ensures temporary or project-specific files (like virtual environments, `.idea`, etc.) are ignored by Git.

## 📦 Requirements

* Python 3.x  
* TensorFlow or PyTorch (depending on your implementation)  
* NumPy  
* Matplotlib (optional, for visualization)

You can install dependencies using:

```bash
pip install -r requirements.txt
(Note: Make sure to create a requirements.txt if you haven’t already.)
```

# 📁 Dataset
To make this program work, you'll need the .npy files from the Quick, Draw! dataset.

```
Download them from the following link:
👉 Google Cloud Storage – Quick Draw Dataset

After downloading, place the .npy files into a folder (e.g., data/) and ensure your scripts reference that directory correctly.
```

#🚀 Getting Started

```Download the .npy files as described above.

Train the model by running:

python train.py
Use the trained model with the doodle recognition script:
python aiDoodle.py
```

# ✍️ License
```
This project is provided for educational and experimental use.
Feel free to use, modify, and explore!
```

# Happy Doodling! 🎨🖌️
