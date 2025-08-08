import pandas as pd
import pickle
import sys
import os

MODEL_PATH = os.path.join('trained_model', 'repd_model_CA.pkl')

# Load trained model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Predict defect likelihood for each class
# Assumes model has a predict_proba or similar method

def predict(csv_path, model_path=MODEL_PATH):
    df = pd.read_csv(csv_path)
    # Drop non-feature columns (project_name, version, class_name, bug)
    feature_cols = [col for col in df.columns if col not in ['project_name', 'version', 'class_name', 'bug']]
    X = df[feature_cols]
    model = load_model(model_path)
    # If model has predict_proba, use it; else use predict
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        # Assume binary classification: defective = probs[:,1]
        df['defect_probability'] = probs[:,1]
    else:
        preds = model.predict(X)
        df['defect_prediction'] = preds
    return df[['class_name', 'defect_probability' if 'defect_probability' in df else 'defect_prediction']]

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python predict_semantic.py <metrics_csv>')
        sys.exit(1)
    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f'CSV file not found: {csv_path}')
        sys.exit(1)
    result_df = predict(csv_path)
    print(result_df.to_string(index=False))
    # Optionally, save to file
    result_df.to_csv('semantic_predictions.csv', index=False)
    print('Predictions saved to semantic_predictions.csv')
