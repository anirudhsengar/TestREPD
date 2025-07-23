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
    df = pd.read_csv(features_file)

    # Store the file names for the final report
    file_names = df["File"].values

    # Prepare the data for prediction by dropping the 'File' column
    X = df.drop(columns=["File"]).values

    # Initialize the model with the same architecture as during training
    # The parameters are: layers=[input_dim, hidden1, hidden2, ...], learning_rate, epochs, batch_size
    # The input dimension is taken from the number of columns in the feature set
    autoencoder = AutoEncoder([X.shape[1], 17, 7], 0.001, 500, 128)
    classifier = REPD(autoencoder)

    # Restore the trained model from the checkpoint file
    saver = tf.train.Saver()
    try:
        saver.restore(autoencoder.sess, "./REPD_Model.ckpt")
    except Exception as e:
        print(f"Error restoring the model: {e}")
        print("Please ensure that the model checkpoint files (REPD_Model.ckpt.*) are in the same directory.")
        sys.exit(1)

    # Make predictions on the new data
    predictions = classifier.predict(X)

    # Print the results
    print("Prediction Results:")
    print("-------------------")
    for i, file_name in enumerate(file_names):
        prediction = "Likely to contain a bug" if predictions[i] == 1 else "Unlikely to contain a bug"
        print(f"File: {file_name} -> {prediction}")

    # Close the TensorFlow session
    autoencoder.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_features.csv>")
        sys.exit(1)
    
    features_csv_path = sys.argv[1]
    predict(features_csv_path)
