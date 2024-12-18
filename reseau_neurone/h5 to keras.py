import os
from tensorflow.keras.models import load_model

# Directory containing the .h5 models
models_dir = 'saved_models'

# Get a list of all .h5 files in the directory
model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]

for model_file in model_files:
    # Construct full file path
    model_path = os.path.join(models_dir, model_file)
    
    # Load the model from the .h5 file
    model = load_model(model_path)
    
    # Create new file name with .keras extension
    new_model_file = os.path.splitext(model_file)[0] + '.keras'
    new_model_path = os.path.join(models_dir, new_model_file)
    
    # Save the model in .keras format
    model.save(new_model_path)
    
    print(f'Converted {model_file} to {new_model_file}')
    
    # Delete the original .h5 file
    os.remove(model_path)
    print(f'Deleted original file {model_file}')