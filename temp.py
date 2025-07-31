import pandas as pd
from REPD_Impl import REPD
from autoencoder import AutoEncoder
import warnings
import tensorflow.compat.v1 as tf

# Suppress warnings
tf.disable_v2_behavior()
warnings.simplefilter("ignore")

def debug_model_attributes():
    """Debug what the REPD model actually contains"""
    
    # Load training data
    print("Loading training data...")
    df_train = pd.read_csv("metrics.csv")
    
    # Prepare training data
    X_train = df_train.drop(columns=["File", "defects"]).values
    y_train = df_train['defects'].values
    
    print("Training model...")
    
    # Initialize and train the model
    autoencoder = AutoEncoder([20, 17, 7], 0.001, 500, 128)
    classifier = REPD(autoencoder)
    classifier.fit(X_train, y_train)
    
    print("Debugging classifier attributes...")
    
    # Check all attributes
    for attr in dir(classifier):
        if not attr.startswith('_'):
            value = getattr(classifier, attr)
            print(f"{attr}: {type(value)} = {value}")
    
    # Specifically check the problematic attributes
    print("\nSpecific attribute analysis:")
    if hasattr(classifier, 'dnd'):
        print(f"dnd: {type(classifier.dnd)} = {classifier.dnd}")
        if hasattr(classifier.dnd, 'args'):
            print(f"dnd.args: {type(classifier.dnd.args)} = {classifier.dnd.args}")
    
    if hasattr(classifier, 'dnd_pa'):
        print(f"dnd_pa: {type(classifier.dnd_pa)} = {classifier.dnd_pa}")
    
    if hasattr(classifier, 'dd'):
        print(f"dd: {type(classifier.dd)} = {classifier.dd}")
        if hasattr(classifier.dd, 'args'):
            print(f"dd.args: {type(classifier.dd.args)} = {classifier.dd.args}")
    
    if hasattr(classifier, 'dd_pa'):
        print(f"dd_pa: {type(classifier.dd_pa)} = {classifier.dd_pa}")
    
    # Close session
    autoencoder.close()

if __name__ == "__main__":
    debug_model_attributes()