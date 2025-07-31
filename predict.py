import pandas as pd
from REPD_Impl import REPD
from autoencoder import AutoEncoder
import warnings
import tensorflow.compat.v1 as tf
import numpy as np
import pickle
import sys
import os
import scipy.stats as st

# Suppress warnings
tf.disable_v2_behavior()
warnings.simplefilter("ignore")

def convert_to_risk_scores(predictions):
    """Convert PDF values to interpretable risk scores (0-100)"""
    risk_scores = []
    
    for pred in predictions:
        defective_pdf = pred[0]
        non_defective_pdf = pred[1]
        
        # Use ratio-based approach for better interpretation
        total = defective_pdf + non_defective_pdf
        if total == 0:
            risk_score = 50  # Neutral when both are 0
        else:
            # Higher defective PDF relative to non-defective = higher risk
            risk_score = (defective_pdf / total) * 100
        
        risk_scores.append({
            'risk_score': risk_score
        })
    
    return risk_scores
    
def format_results(file_names, risk_data):
    """Format results with interpretable risk scores"""
    output = ["🎯 Code Quality Risk Assessment\n"]
    
    for i, file_name in enumerate(file_names):
        risk_score = risk_data[i]['risk_score']
        
        output.append(f"File: {file_name}")
        output.append(f"- Risk Score: {float(risk_score):.1f}/100")
    
    return "\n".join(output)

def recreate_distribution(dist_name, params):
    """Recreate a scipy distribution from saved name and parameters"""
    if dist_name is None:
        return None
    
    try:
        dist_class = getattr(st, dist_name)
        if params:
            return dist_class(*params)
        else:
            return dist_class()
    except Exception as e:
        print(f"Warning: Could not recreate distribution {dist_name}: {e}")
        return None

def load_trained_model(model_dir="trained_model"):
    """Load the pre-trained model"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Trained model not found at {model_dir}. Please ensure the model is trained and saved.")
    
    # Load metadata
    metadata_path = os.path.join(model_dir, "metadata.pkl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Load REPD classifier parameters
    classifier_params_path = os.path.join(model_dir, "classifier_params.pkl")
    if not os.path.exists(classifier_params_path):
        raise FileNotFoundError(f"Classifier parameters not found at {classifier_params_path}")
    
    with open(classifier_params_path, 'rb') as f:
        classifier_params = pickle.load(f)
    
    # Recreate the autoencoder with saved architecture
    autoencoder = AutoEncoder(
        metadata['architecture'], 
        metadata['learning_rate'], 
        metadata['epochs'], 
        metadata['batch_size']
    )
    
    # Load the saved autoencoder weights
    autoencoder_path = os.path.join(model_dir, "autoencoder")
    autoencoder.load(autoencoder_path)
    
    # Recreate REPD classifier
    classifier = REPD(autoencoder)
    
    # Recreate distributions from saved names and parameters
    classifier.dnd = recreate_distribution(
        classifier_params.get('dnd_name'), 
        classifier_params.get('dnd_params')
    )
    classifier.dnd_pa = classifier_params.get('dnd_pa')
    
    classifier.dd = recreate_distribution(
        classifier_params.get('dd_name'), 
        classifier_params.get('dd_params')
    )
    classifier.dd_pa = classifier_params.get('dd_pa')
    
    return classifier

def predict(features_file, model_dir="trained_model"):
    """Make predictions using pre-trained model"""
    
    print("Loading pre-trained model...")
    classifier = load_trained_model(model_dir)
    
    # Load test data
    df_test = pd.read_csv(features_file)
    file_names = df_test["File"].values
    X_test = df_test.drop(columns=["File"]).values
        
    print("Making predictions...")
        
    # Make predictions (PDF values)
    pdf_predictions = classifier.predict(X_test)
    
    # Convert to interpretable risk scores
    risk_data = convert_to_risk_scores(pdf_predictions)

    # Format and print results
    print(format_results(file_names, risk_data))
    
    # Close the session
    classifier.dim_reduction_model.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_features.csv>")
        print("Make sure the trained model exists in the 'trained_model' directory.")
        sys.exit(1)
    
    features_csv_path = sys.argv[1]
    predict(features_csv_path)