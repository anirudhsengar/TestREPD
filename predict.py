import pandas as pd
from REPD_Impl import REPD
from autoencoder import AutoEncoder
import warnings
import tensorflow.compat.v1 as tf
import sys

# Suppress warnings
tf.disable_v2_behavior()
warnings.simplefilter("ignore")

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

    print(X_test)

    # Initialize the model with the same architecture as during training
    # The parameters are: layers=[input_dim, hidden1, hidden2, ...], learning_rate, epochs, batch_size
    # The input dimension is taken from the number of columns in the feature set
    autoencoder = AutoEncoder([20, 17, 7], 0.001, 500, 128)
    classifier = REPD(autoencoder)
    classifier.fit(X_train, y_train)

    # Make predictions on the new data
    predictions = classifier.predict(X_test)

    # Print the results
    print("Prediction Results:")
    print("-------------------")
    for i, file_name in enumerate(file_names):
        # PDF is the Probability Density Function
        print(f"File: {file_name} -> PDF(Defective) = {predictions[i][0] * 100000}, PDF(Non-defective) = {predictions[i][1] * 100000}")

    # Close the TensorFlow session
    autoencoder.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_features.csv>")
        sys.exit(1)
    
    features_csv_path = sys.argv[1]
    predict(features_csv_path)
