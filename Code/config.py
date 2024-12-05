from pathlib import Path
import os

def get_config():
    return {
        "vocab_size": 8192,
        "batch_size": 16,
        "num_epochs": 30,
        "lr": 10**-4,
        "seq_len": 256,
        "d_model": 512,
        "dropout": 0.1,
        "N": 6,
        "h": 8,
        "d_ff": 2048,
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "model_",
        "output_file": "bleu_scores_base_30.txt",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
    }

def get_weights_file_path(config, epoch: int):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}_{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def manage_saved_models(config, current_epoch):
    model_folder = config['model_folder']
    
    # Ensure the model folder exists
    if not os.path.exists(model_folder):
        print(f"Model folder {model_folder} does not exist. No models to manage.")
        return

    # List all .pt files in the model folder
    model_files = [f for f in os.listdir(model_folder)]
    
    # Sort the files based on the epoch number in the filename
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # If we have more than 2 model files, remove the oldest ones
    while len(model_files) > 2:
        oldest_file = model_files.pop(0)
        file_path = os.path.join(model_folder, oldest_file)
        try:
            os.remove(file_path)
            print(f"Removed old model checkpoint: {oldest_file}")
        except OSError as e:
            print(f"Error removing file {oldest_file}: {e}")

    # Print remaining files for debugging
    print(f"Remaining model files: {model_files}")
