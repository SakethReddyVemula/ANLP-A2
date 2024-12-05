import torch
from utils import get_config, get_dataset, get_weights_file_path, generate_bleu_scores_file, greedy_decode
from train import get_model, run_validation
import torch.optim as optim
import torch.nn as nn
from torchmetrics.text import BLEUScore
import torch.optim as optim
from tqdm import tqdm
from rouge_score import rouge_scorer

def test_model(config):
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(config)
    train_dataloader, dev_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model_filename = get_weights_file_path(config, f"{(int(config['num_epochs']) - 1):02d}")
    state = torch.load(model_filename, map_location=device)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    optimizer = optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    del state

    model.eval()

    # Initialize metrics
    bleu_metric = BLEUScore()
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Lists to store results
    source_texts = []
    expected = []
    expected_list = []
    predicted = []
    
    console_width = 80
    
    print("\nRunning test evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            # Perform greedy decoding
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
            
            # Get text representations
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            # Store results
            source_texts.append(source_text)
            target_text_list = [target_text]
            expected_list.append(target_text_list)
            expected.append(target_text)
            predicted.append(model_out_text)
    
    # Calculate metrics
    bleu = bleu_metric(predicted, expected_list)
    
    # Calculate ROUGE scores
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for ref, pred in zip(expected, predicted):
        scores = rouge_scorer_obj.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
    
    # Average ROUGE scores
    for key in rouge_scores:
        rouge_scores[key] /= len(expected)
    
    # Print test results
    print('-' * console_width)
    print(f"Test Results:")
    print(f"Number of test examples: {len(predicted)}")
    print(f"BLEU score: {bleu:.4f}")
    print(f"ROUGE1 score: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE2 score: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGEL score: {rouge_scores['rougeL']:.4f}")
    
    # Print some example translations
    print("\nExample Translations:")
    for i in range(min(5, len(predicted))):
        print('-' * console_width)
        print(f'SOURCE: {source_texts[i]}')
        print(f'TARGET: {expected[i]}')
        print(f'PREDICTED: {predicted[i]}')

    generate_bleu_scores_file(model, test_dataloader, tokenizer_src, tokenizer_tgt, max_len=config["seq_len"], device=device, output_file=config["output_file"])

if __name__ == "__main__":
    config = get_config()
    test_model(config)
    
    
