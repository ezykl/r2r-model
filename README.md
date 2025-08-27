# R2R Model – Image Classification API

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-green?logo=flask&logoColor=white)

This repository contains the **image classification model** for the **R2R mobile app**.  
It is trained on **180 classes of items** (tools, equipment, and products) and is integrated into the R2R ecosystem.

The model serves as an intelligent classification system that can identify various items and automatically categorize them as **Accepted** or **Prohibited** based on predefined business rules.

---

## 🚀 Features

- **180-class image classification** with high accuracy
- **Smart confidence-based predictions** (Top-1, Top-3, or Unknown)
- **Flask RESTful API** for seamless integration
- **Automatic item categorization** (Accepted vs Prohibited)
- **Multi-format image support** (JPEG, PNG, GIF)
- **Real-time inference** optimized for production use
- **Comprehensive error handling** and validation

---

## 📂 Repository Structure

```
r2r-model/
│
├── model/
│   └── r2r_model.keras          # Trained Keras model
│
├── app.py                       # Flask API server
├── requirements.txt             # Python dependencies
├── classes.md                   # Complete class documentation
├── README.md                    # Project documentation
├── .gitattributes               # Git attributes
└── .gitignore                   # Git ignore rules
```

---

## 🧠 Model Architecture & Performance

- **Base Architecture**: MobileNet (transfer learning)
- **Input Size**: 224x224 RGB images
- **Classes**: 180 distinct categories
- **Framework**: TensorFlow/Keras
- **Model Type**: Fine-tuned convolutional neural network
- **Preprocessing**: Automatic normalization and resizing
- **Inference Time**: ~100ms per image (CPU)
- **Model Size**: Optimized for mobile deployment

### 🎯 API Behavior

The API provides intelligent item classification with confidence-based filtering:

| Confidence Level  | Behavior          | Output                                |
| ----------------- | ----------------- | ------------------------------------- |
| **≥ 70%**         | High confidence   | **Top-1 prediction** with category    |
| **≥ 20% & < 70%** | Medium confidence | **Top-3 predictions** with categories |
| **< 20%**         | Low confidence    | **"Unknown"** classification          |

Each prediction includes:

- **Predicted Item**: Class name
- **Category**: "Accepted" or "Prohibited"
- **Confidence**: Percentage score

---

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/ezykl/r2r-model.git
   cd r2r-model
   ```

2. **Create virtual environment** (recommended)

   ```bash
   python -m venv env

   # Windows
   .\env\Scripts\activate

   # macOS/Linux
   source env/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Flask server**

   ```bash
   python app.py
   ```

5. **Verify installation**
   ```bash
   curl http://localhost:5000/
   # Expected: {"message": "API is running successfully!"}
   ```

---

## 📡 API Usage

### Endpoints

#### `GET /`

Health check endpoint

```json
{
  "message": "API is running successfully!"
}
```

#### `POST /predict`

Image classification endpoint

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file (key: `image`)
- Supported formats: JPEG, JPG, PNG, GIF

**Response Examples:**

**High Confidence (≥70%):**

```json
[
  {
    "Predicted Item": "Cordless Drill",
    "Category": "Accepted",
    "Confidence": "87.45%"
  }
]
```

**Medium Confidence (20-70%):**

```json
[
  {
    "Predicted Item": "Hammer Drill",
    "Category": "Accepted",
    "Confidence": "45.23%"
  },
  {
    "Predicted Item": "Impact Wrench",
    "Category": "Accepted",
    "Confidence": "32.15%"
  },
  {
    "Predicted Item": "Cordless Drill",
    "Category": "Accepted",
    "Confidence": "22.67%"
  }
]
```

**Low Confidence (<20%):**

```json
[
  {
    "Predicted Item": "Unknown",
    "Category": "N/A",
    "Confidence": "15.34% (Low Confidence)"
  }
]
```

### Testing with cURL

```bash
# Test with an image file
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict
```

### Testing with Python

```python
import requests

url = "http://localhost:5000/predict"
files = {"image": open("your_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

## 🏷️ Classes & Categories

The model recognizes **180 distinct classes** across various categories:

- **Tools & Equipment**: Drills, hammers, wrenches, saws, etc.
- **Electronics**: Cameras, laptops, audio equipment, etc.
- **Camping & Outdoor**: Tents, chairs, portable stoves, etc.
- **Automotive**: Car jacks, battery chargers, etc.
- **Construction**: Concrete mixers, scaffolds, welding equipment, etc.

**📋 Complete Class List**: See [`classes.md`](classes.md) for all 180 classes.

### Prohibited vs Accepted Items

The system automatically categorizes items based on business rules:

**Prohibited Items** (35 classes):

- Firearms, explosives, hazardous materials
- Food, beverages, prescription drugs
- Personal items (clothing, jewelry, pets)
- Vehicles, real estate, identity documents

**Accepted Items** (145 classes):

- Tools, equipment, electronics
- Camping gear, sports equipment
- Professional instruments, safety gear

---

## 🛠️ Development

### Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: No image uploaded, invalid format
- **500 Internal Server Error**: Processing errors with detailed messages

### Environment Variables

```bash
export TF_ENABLE_ONEDNN_OPTS=0  # Disable oneDNN optimizations
```

### Model Requirements

- **Input**: 224x224 RGB images
- **Preprocessing**: Automatic resizing and normalization (0-1)
- **Format**: Keras (.keras) model file

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

```
Copyright (c) 2025 Ezekiel Villadolid
```

---

## 👨‍💻 Author

**Ezekiel Villadolid**

- GitHub: [@ezykl](https://github.com/ezykl)

---

## 🙏 Acknowledgments

- Built with TensorFlow and Flask
- Designed for the R2R mobile application ecosystem
- Optimized for real-time item classification and categorization

---

_For questions or support, please open an issue on GitHub._
