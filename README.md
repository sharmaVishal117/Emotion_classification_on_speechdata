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
The system recognizes 8 distinct emotional states:

Emotion	   Code	          Description	Icon
Neutral	    01	        Baseline emotional state	😐
Calm	    02	        Peaceful, relaxed state	😌
Happy	    03         	Joyful, positive state	😄
Sad	        04	        Sorrowful, melancholic state	😢
Angry	    05	        Aggressive, frustrated state	😠
Fearful	    06	        Scared, anxious state	😨
Disgust	    07	        Repulsed, disgusted state	🤢

🏗 System Architecture

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

## �🚀 Features

- **Real-time Emotion Detection**: Upload audio files and get instant emotion predictions
- **Multiple Audio Formats**: Supports WAV, MP3, FLAC, and M4A files (local version)
- **Visual Analysis**: Interactive MFCC feature visualization and confidence scoring
- **User-friendly Interface**: Clean, modern UI with detailed explanations
- **High Accuracy**: Deep CNN model with SE-Blocks for robust emotion recognition
- **Demo Mode**: Cloud deployment shows sample predictions without audio processing

## 🎯 Supported Emotions

- Neutral
- Calm  
- Happy
- Sad
- Angry
- Fearful
- Disgust

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: TensorFlow/Keras
- **Audio Processing**: Librosa (local version)
- **Feature Extraction**: MFCC (Mel-Frequency Cepstral Coefficients)
- **Visualization**: Matplotlib, Seaborn

## 📊 Model Architecture

- Deep Convolutional Neural Network with SE-Blocks
- 60 MFCC coefficients as input features
- Processes 3-second audio segments at 22,050 Hz sample rate
- Custom focal loss for handling class imbalance

## 🚀 Quick Start

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

This project is open source and available under the MIT License.
