# MARS Open Projects 2025 - Project 1: Speech Emotion Recognition System
🎯 Project Overview
This project implements a sophisticated Speech Emotion Recognition (SER) system using deep learning techniques as part of the MARS Open Projects 2025 initiative. The system employs a hybrid CNN + SE blocks architecture trained on the RAVDESS dataset to classify emotions in speech and song audio files with high accuracy.

📋 Project Requirements
-Primary Objective: Develop an AI system that can accurately identify emotions from speech audio
-Performance Targets:
-Weighted F1 Score > 80%
-Overall Accuracy > 80%
-Individual Class Recalls > 75%
-Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
-Implementation: Complete end-to-end pipeline with web interface

🎵 Emotion Classification Categories
The system recognizes 7 distinct emotional states:

| Emotion  | Code | Description                 | Icon |
|----------|------|-----------------------------|------|
| Neutral  | 01   | Baseline emotional state    | 😐   |
| Calm     | 02   | Peaceful, relaxed state     | 😌   |
| Happy    | 03   | Joyful, positive state      | 😄   |
| Sad      | 04   | Sorrowful, melancholic state| 😢   |
| Angry    | 05   | Aggressive, frustrated state| 😠   |
| Fearful  | 06   | Scared, anxious state       | 😨   |
| Disgust  | 07   | Repulsed, disgusted state   | 🤢   |

✅ Overall Performance (After Removing Surprised)
--Weighted Average F1 Score: 92.00% 
--Macro Average F1 Score: 93.11% 
--Overall Accuracy: 92.20% 
--Test Precision: 94.00% 


🏗 **System Architecture**

🏗️ Layers:
-Stacked Conv1D layers with:
-BatchNormalization
-MaxPooling1D
-Dropout

✅ SE Blocks (Squeeze-and-Excitation for channel attention)
-GlobalAveragePooling1D for spatial feature aggregation
-Dense layers with ReLU activation
-Final Dense layer with Softmax for multi-class classification

🎧 Feature Engineering
-Audio Processing: 3-second segments at 22.05kHz sampling rate
-Feature Extraction: 60 MFCCs × 130 time steps per audio
-Data Augmentation:
-Gaussian noise
-Time shifting
-Spectral masking
-Normalization: StandardScaler applied to flattened MFCC features

⚙️ Loss Function & Optimization
-Loss: Categorical Focal Loss (gamma=2.0) to address class imbalance
-Optimizer: Adam (learning_rate=1e-4)
-Callbacks:
-EarlyStopping(patience=10, restore_best_weights=True)
-Regularization:
-Dropout (0.3–0.4)
-BatchNormalization
-SE channel recalibration (via SE Blocks)

🏆 **Performance Achievement**

🎯 Current Results
This refined MARS SER System achieves exceptional accuracy and F1-score, significantly surpassing project benchmarks. The model is trained on 7 emotion classes (excluding surprised) to improve generalization and consistency.

| Metric                | Target | Achieved    |
| --------------------- | ------ | ------------ |
| **Weighted F1 Score** | > 80%  | 92.00%   |
| **Overall Accuracy**  | > 80%  | 92.20%   |
| **Per-Class Recalls** | > 75%  | Most > 85% |

 Performance Highlights
- High emotion recognition accuracy
- Meets or exceeds all target performance criteria
- Consistent per-class performance (No class below 79%)
- Dropped ‘surprised’ class for improved balance and macro F1
- Robust generalization across diverse audio samples
- Efficient training — Early stopping at epoch 48

| Metric              | Score      |
| ------------------- | ---------- |
| ✅ Weighted F1 Score | **92.00%** |
| ✅ Overall Accuracy  | **92.20%** |
| ✅ Macro F1 Score    | **93.11%** |
| ✅ Test Precision    | **94.00%** |
| ✅ Test Recall       | **92.00%** |

| Emotion | Precision | Recall | F1-Score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Angry   | 1.00      | 0.87   | 0.93     | 301     |
| Calm    | 0.96      | 0.99   | 0.97     | 301     |
| Disgust | 0.93      | 1.00   | 0.97     | 153     |
| Fearful | 0.76      | 1.00   | 0.86     | 301     |
| Happy   | 0.95      | 0.88   | 0.91     | 301     |
| Neutral | 1.00      | 1.00   | 1.00     | 150     |
| Sad     | 0.96      | 0.79   | 0.87     | 301     |

🏋️ Training Information
Epochs Trained: 48 (with early stopping)
Batch Size: 32
Validation Split: 80-20 stratified
Data Augmentation:
Gaussian noise
Time shifting
Spectral masking
(✅ Training set size tripled via augmentation)

**Before Removing Surprised Class**
| Emotion   | Precision | Recall | F1-Score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Angry     | 1.00      | 0.84   | 0.92     | 301     |
| Calm      | 0.97      | 0.80   | 0.88     | 301     |
| Disgust   | 0.97      | 0.91   | 0.94     | 153     |
| Fearful   | 0.95      | 0.80   | 0.87     | 301     |
| Happy     | 0.89      | 0.71   | 0.79     | 301     |
| Neutral   | 0.76      | 0.93   | 0.84     | 150     |
| Sad       | 0.97      | 0.91   | 0.94     | 301     |
| Surprised | 0.44      | 1.00   | 0.61     | 154     |

| Metric          | Score  |
| --------------- | ------ |
| Accuracy        | 84.20% |
| Macro Avg F1    | 84.37% |
| Weighted Avg F1 | 86.00% |


### Local Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd emotion_classification_on_speechdata
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
streamlit run app.py
```
### Streamlit Cloud Deployment

This app is ready for deployment on Streamlit Cloud. Simply:

1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Deploy with one click!

**Dependencies   
| Package      | Version |
| ------------ | ------- |
| streamlit    | 1.34.0  |
| numpy        | 1.24.3  |
| librosa      | 0.10.1  |
| tensorflow   | 2.12.0  |
| scikit-learn | 1.2.2   |
| matplotlib   | 3.7.1   |
| seaborn      | 0.12.2  |
| pandas       | 1.5.3   |



## 📁 Project Structure

```
emotion_classification_on_speechdata/
├── app.py                 # Main Streamlit application
├── model/                 # Pre-trained model files
│   ├── emotion_model (2).h5
│   ├── label_encoder (3).pkl
│   └── scaler (2).pkl
├── requirements.txt       # Python dependencies
├── packages.txt          # System packages for Streamlit Cloud
├── runtime.txt           # Python version specification
└── .streamlit/
    └── config.toml       # Streamlit configuration
```
## 🎵 Usage

1. **Upload Audio**: Choose an audio file in supported formats
2. **View Analysis**: See file information and audio player
3. **Get Predictions**: View detected emotion with confidence scores
4. **Explore Features**: Examine MFCC visualizations and audio characteristics

## 📋 Requirements

See `requirements.txt` for detailed Python package requirements.

## 🔧 Configuration

The app includes optimized configuration for Streamlit Cloud deployment:
- Maximum upload size: 200MB
- Custom theme with professional styling
- Efficient caching for model loading

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and Dataset: RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song Framework: TensorFlow/Keras
