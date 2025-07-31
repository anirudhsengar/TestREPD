import pandas as pd
from REPD_Impl import REPD
from autoencoder import AutoEncoder
import warnings
import tensorflow.compat.v1 as tf
import numpy as np
import sys

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
            'risk_score': risk_score,
            'confidence': max(defective_pdf, non_defective_pdf) / total if total > 0 else 0
        })
    
    return risk_scores
    
def format_results(file_names, risk_data):
    """Format results with interpretable risk scores"""
    output = ["## 🎯 Code Quality Risk Assessment\n"]
    
    for i, file_name in enumerate(file_names):
        risk_score = risk_data[i]['risk_score']
        confidence = risk_data[i]['confidence']
        
        output.append(f"File: {file_name}")
        output.append(f"- Risk Score: {float(risk_score):.1f}/100")
        output.append(f"- Model Confidence: {float(confidence):.1%}")
    
    return "\n".join(output)

def predict(features_file):
    # Load the dataset from the provided CSV file
    df_train = pd.read_csv("metrics.csv")
    df_test = pd.read_csv(features_file)

    # Store the file names for the final report
    file_names = df_test["File"].values

    # Prepare the data for prediction by dropping the 'File' column
    X_test = df_test.drop(columns=["File"]).values

    X_train = df_train.drop(columns=["File", "defects"]).values
    y_train = df_train['defects'].values

    # Initialize the model with the same architecture as during training
    # The parameters are: layers=[input_dim, hidden1, hidden2, ...], learning_rate, epochs, batch_size
    # The input dimension is taken from the number of columns in the feature set
    autoencoder = AutoEncoder([20, 17, 7], 0.001, 500, 128)
    classifier = REPD(autoencoder)
    classifier.fit(X_train, y_train)

    # # Make predictions on the new data
    # predictions = classifier.predict(X_test)

    # factor = pow(10, 4)
    # # Print the results
    # print("Prediction Results:")
    # print("-------------------")
    # for i, file_name in enumerate(file_names):
    #     # PDF is the Probability Density Function
    #     print(f"File: {file_name}\n P(Defective | Reconstruction Error) = {predictions[i][0] * factor}\n P(Non-defective | Reconstruction Error) = {predictions[i][1] * factor}")
    # # Close the TensorFlow session
    # autoencoder.close()

    # Make predictions (PDF values)
    pdf_predictions = classifier.predict(X_test)
    
    # Convert to interpretable risk scores
    risk_data = convert_to_risk_scores(pdf_predictions)
    
    # Format and print results
    print(format_results(file_names, risk_data))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_features.csv>")
        sys.exit(1)
    
    features_csv_path = sys.argv[1]
    predict(features_csv_path)
