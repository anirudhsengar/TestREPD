import pandas as pd
from REPD_Impl import REPD
from autoencoder import AutoEncoder
import warnings
import tensorflow.compat.v1 as tf
import json
import os
import numpy as np

# Suppress warnings
tf.disable_v2_behavior()
warnings.simplefilter("ignore")

def extract_distribution_info(dist, params_tuple):
    """Extract distribution name and parameters from scipy distribution"""
    if dist is None:
        return None, []
    
    try:
        # Get distribution name
        if hasattr(dist, 'name'):
            dist_name = dist.name
        else:
            print(f"Unknown distribution type: {type(dist)}")
            return None, []
        
        # Use the fitted parameters from the params_tuple
        if params_tuple is not None:
            params = [float(p) for p in params_tuple]  # Convert numpy types to float
        else:
            params = []
        
        print(f"Extracted distribution: {dist_name} with {len(params)} parameters: {params}")
        return dist_name, params
        
    except Exception as e:
        print(f"Error extracting distribution info: {e}")
        return None, []

def convert_to_json_serializable(obj):
    """Convert numpy/complex objects to JSON serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        # Try to convert to string as fallback
        return str(obj)

def train_and_save_model(training_data_path="metrics.csv", model_save_dir="trained_model"):
    """Train the REPD model and save it for later use"""
    
    # Remove existing model directory to ensure clean save
    if os.path.exists(model_save_dir):
        import shutil
        shutil.rmtree(model_save_dir)
    
    # Create model directory
    os.makedirs(model_save_dir)
    
    # Load training data
    print("Loading training data...")
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
    
    # Save the autoencoder weights
    autoencoder_save_path = os.path.join(model_save_dir, "autoencoder")
    autoencoder.save(autoencoder_save_path)
    
    # Debug the distributions before extraction
    print("\nDebugging distributions:")
    print(f"classifier.dnd: {type(classifier.dnd)} = {classifier.dnd}")
    print(f"classifier.dd: {type(classifier.dd)} = {classifier.dd}")
    
    if hasattr(classifier, 'dnd_pa'):
        print(f"classifier.dnd_pa: {type(classifier.dnd_pa)} = {classifier.dnd_pa}")
    if hasattr(classifier, 'dd_pa'):
        print(f"classifier.dd_pa: {type(classifier.dd_pa)} = {classifier.dd_pa}")
    
    # Extract distribution information WITH their fitted parameters
    print("\nExtracting distribution info...")
    dnd_name, dnd_params = extract_distribution_info(classifier.dnd, getattr(classifier, 'dnd_pa', None))
    dd_name, dd_params = extract_distribution_info(classifier.dd, getattr(classifier, 'dd_pa', None))
    
    # Save classifier parameters as JSON
    classifier_params = {
        'dnd_name': dnd_name,
        'dnd_params': dnd_params,  # These now contain the actual fitted parameters
        'dd_name': dd_name,
        'dd_params': dd_params     # These now contain the actual fitted parameters
    }
    
    print(f"\nSaving classifier params: {classifier_params}")
    
    # Save as JSON
    with open(os.path.join(model_save_dir, "classifier_params.json"), 'w') as f:
        json.dump(classifier_params, f, indent=2)
    
    # Save training metadata as JSON
    metadata = {
        'input_shape': int(X_train.shape[1]),
        'architecture': [20, 17, 7],
        'learning_rate': 0.001,
        'epochs': 500,
        'batch_size': 128
    }
    
    with open(os.path.join(model_save_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_save_dir}")
    print("Training completed!")
    
    # Close TensorFlow session
    autoencoder.close()

if __name__ == "__main__":
    train_and_save_model()