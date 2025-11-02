# Skin Disease AI
## AI-Powered Skin Lesion Diagnosis System

### 🎯 Overview

An advanced machine learning solution for automated skin disease classification and diagnosis. This system leverages deep learning to assist healthcare professionals and individuals in identifying common skin conditions through image analysis.

---

### 📊 Dataset Composition

The model is trained on a comprehensive dataset of **1,657 clinical images** spanning **7 distinct classes**:

| Class | Description |
|-------|-------------|
| **Acne** | Inflammatory skin condition affecting hair follicles |
| **Carcinoma** | Malignant skin cancer lesions |
| **Eczema** | Chronic inflammatory skin disorder |
| **Keratosis** | Benign skin growths and lesions |
| **Millia** | Small keratin-filled cysts |
| **Rosacea** | Chronic facial skin condition |
| **Non-Lesion** | Control class for healthy skin |

**Data Sources:**
- Public dermatological image repositories
- Curated clinical datasets
- Proprietary collected images

**Preprocessing & Augmentation:**
- Class imbalance mitigation through strategic data augmentation
- Standardized image preprocessing pipeline
- Train/validation/test split for robust model evaluation

---

### 🧠 Model Architecture

**Framework:** Xception (Extreme Inception) Deep Learning Architecture

**Key Features:**
- Transfer learning from ImageNet pre-trained weights
- Depthwise separable convolutions for efficient feature extraction
- Fine-tuned for dermatological image classification

**Performance Metrics:**
- **Test Accuracy:** 92%
- Robust classification across all skin lesion categories
- Validated on hold-out test set

---

### 📁 Project Structure
```
skin-disease-ai/
│
├── preprocessing.py           # Data loading and preprocessing pipeline
├── sets_visualization.py      # Dataset distribution analysis
├── augmentation.py           # Data augmentation for class balancing
├── model.py                  # Xception model implementation
├── evaluate.py               # Model evaluation and metrics
├── predict.py                # Batch inference utility
├── skin_disease_app.py       # Streamlit web application
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

#### Module Descriptions

**`preprocessing.py`**
- Loads raw image data from source directories
- Applies standardization and normalization
- Generates stratified train/validation/test splits

**`sets_visualization.py`**
- Visualizes class distribution across datasets
- Generates statistical reports on data composition

**`augmentation.py`**
- Implements augmentation strategies (rotation, flip, zoom, etc.)
- Balances underrepresented classes
- Expands training dataset diversity

**`model.py`**
- Defines Xception-based neural network architecture
- Configures training hyperparameters
- Implements model compilation and training loops

**`evaluate.py`**
- Generates confusion matrix visualizations
- Plots training/validation accuracy and loss curves
- Produces detailed classification reports (precision, recall, F1-score)

**`predict.py`**
- Batch prediction interface for multiple images
- Outputs classification results with confidence scores

**`skin_disease_app.py`**
- Interactive Streamlit web interface
- Real-time image upload and diagnosis
- User-friendly visualization of results

---

### 🚀 Quick Start

#### Prerequisites
- Python 3.8+
- pip package manager

#### Installation

1. Install dependencies:
```bash
   pip install -r requirements.txt
```

#### Running the Application

**Launch the Streamlit Web App:**
```bash
streamlit run skin_disease_app.py
```

The application will open in your default browser at `http://localhost:8501`

**Features:**
- Upload skin lesion images for instant diagnosis
- View classification results with confidence percentages
- Clean, professional interface optimized for clinical use

---

### 🔬 Model Training & Evaluation

**Train the model:**
```bash
python model.py
```

**Evaluate performance:**
```bash
python evaluate.py
```

**Run batch predictions:**
```bash
python predict.py
```

---

### 📈 Results & Performance

- **Overall Accuracy:** 92%
- Robust performance across diverse skin tones and image qualities

---

### ⚠️ Clinical Disclaimer

This system is designed as a **diagnostic aid tool** and should not replace professional medical evaluation. Always consult qualified dermatologists for definitive diagnosis and treatment recommendations.

---

### 🛠️ Technology Stack

- **Deep Learning:** TensorFlow/Keras
- **Computer Vision:** OpenCV
- **Web Framework:** Streamlit
- **Data Science:** NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn

---

### 👨‍💻 Developer

**Akshay Gurav**