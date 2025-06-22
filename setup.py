#!/usr/bin/env python3

import subprocess
import sys
import os

def install_requirements():
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False
    return True

def check_model_files():
    model_files = [
        "model/emotion_model (2).h5",
        "model/scaler (2).pkl", 
        "model/label_encoder (3).pkl"    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nMake sure to:")
        print("   1. Train your model using the full.py script")
        print("   2. Copy the generated model files to the 'model' directory")
        return False
    else:
        print("All model files found!")
        return True

def run_streamlit():
    print("Starting Streamlit app...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\nApp stopped by user")
    except Exception as e:
        print(f"Error running app: {e}")

def main():
    print("Emotion Recognition App Setup")
    print("=" * 40)
    
    if not os.path.exists("app.py"):
        print("app.py not found. Make sure you're in the correct directory.")
        return
    
    if not install_requirements():
        return
    
    if not check_model_files():
        print("\nModel files missing. Please train your model first.")
        return
    
    print("\nSetup complete! Ready to run the app.")
    
    response = input("\nDo you want to start the Streamlit app now? (y/n): ")
    if response.lower() in ['y', 'yes']:
        run_streamlit()
    else:
        print("\nTo run the app later, use: streamlit run app.py")

if __name__ == "__main__":
    main()
