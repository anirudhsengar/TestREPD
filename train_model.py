import pandas as pd
from REPD_Impl import REPD
from autoencoder import AutoEncoder
import warnings
import tensorflow.compat.v1 as tf
import os
import joblib
import numpy as np

# Suppress warnings
tf.disable_v2_behavior()
warnings.simplefilter("ignore")

def train_and_save_model(training_data_path="metrics.csv", model_save_dir="trained_model"):
    """Train the REPD model and save it for later use"""
    
    # Create model directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    # Load training data
    df_train = pd.read_csv(training_data_path)
    
    # Prepare training data
    X_train = df_train.drop(columns=["File", "defects"]).values
    y_train = df_train['defects'].values
    
    print("Training model...")
    
    # Initialize and train the model
    autoencoder = AutoEncoder([20, 17, 7], 0.001, 500, 128)
    classifier = REPD(autoencoder)
    classifier.fit(X_train, y_train)
    
    print("Saving model...")
    
    # Save the autoencoder weights and architecture
    autoencoder_save_path = os.path.join(model_save_dir, "autoencoder")
    autoencoder.save(autoencoder_save_path)
    
    # Save classifier parameters (non-TensorFlow parts)
    classifier_params = {
        'trained': True,
        # Add any other non-TensorFlow parameters your REPD class has
    }
    
    with open(os.path.join(model_save_dir, "classifier_params.pkl"), 'wb') as f:
        joblib.dump(classifier_params, f)
    
    # Save training metadata
    metadata = {
        'input_shape': X_train.shape[1],
        'architecture': [20, 17, 7],
        'learning_rate': 0.001,
        'epochs': 500,
        'batch_size': 128
    }
    
    with open(os.path.join(model_save_dir, "metadata.pkl"), 'wb') as f:
        joblib.dump(metadata, f)
    
    print(f"Model saved to {model_save_dir}")
    print("Training completed!")
    
    # Close TensorFlow session
    autoencoder.close()

if __name__ == "__main__":
    train_and_save_model()