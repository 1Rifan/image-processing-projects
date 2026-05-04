
---

## ⚙️ Technologies Used

- Python
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-image

---

## 🧪 Part 1 — Image Segmentation

### 🔹 1.1 Thresholding (Classical Method)
- Converts grayscale image into binary image
- Uses a fixed threshold value

**Pros:**
- Fast and simple

**Cons:**
- Fails under uneven lighting or overlapping intensities

---

### 🔹 1.2 K-means Clustering (ML Method)
- Unsupervised learning algorithm
- Groups pixels based on similarity

**Pros:**
- Flexible and works on complex images

**Cons:**
- Requires choosing number of clusters (K)

---

## 📊 Comparison

| Method        | Strengths                  | Limitations |
|--------------|---------------------------|-------------|
| Thresholding  | Simple & fast             | Sensitive to lighting |
| K-means       | Flexible segmentation     | Needs K selection |

---

## 🧠 Part 2 — HOG Feature Descriptor

### 🔹 What is HOG?
Histogram of Oriented Gradients (HOG) is a feature descriptor used to capture:
- Shape
- Edges
- Gradient direction

---

### 🔹 Why HOG?
- Robust to lighting changes
- Good for shape-based recognition
- Widely used in object detection

---

## 🔄 Robustness Analysis

The project tests HOG under:
- Rotation (45°)
- Translation (shift)

### Observation:
- Stable under translation
- Sensitive to rotation due to gradient direction changes

---

## ▶️ How to Run

### Install dependencies:
```bash
pip install opencv-python numpy matplotlib scikit-image
``` id="run1"

### Run notebook:
Open in Google Colab and execute cells step by step.

---

## 📌 Notes

- Images are uploaded using Google Colab upload function:
```python
from google.colab import files
uploaded = files.upload()
``` id="note1"

- Make sure image names match:
image1.jpg
image2.jpg


---

## 🎯 Learning Outcomes

This project helps understand:
- Image segmentation techniques
- Difference between classical and ML-based methods
- Feature extraction using HOG
- Effect of transformations on features

---

## 👤 Author

Rifan A.