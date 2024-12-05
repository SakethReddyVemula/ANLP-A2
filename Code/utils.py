import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from pathlib import Path
import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.text import BLEUScore
import torch.optim as optim
from tqdm import tqdm
from rouge_score import rouge_scorer
import sys


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        # In embedding layers, we multiply the weights by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # We need to create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model) # Initialize with zeros

        # Formulas: 
        # PE(pos, 2i) = sin(pos/(10000 ^ (2i/d_model)))
        # PE(pos, 2i + 1) = cos(pos(pos/(10000) ^ (2i/d_model)))
        # For numerical stability we use log instead of exponential

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # Shape: (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # Shape: (1, seq_len, d_model)

        # Register_buffer: for saving tensor inside a model (not as a learned parameter, just for saving when model is saved)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Ensure pe are not learned which training. Since they are always fixed
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float=1.0e-6) -> None:
        super().__init__()
        self.eps = eps # To handle division by zero while normalizing. Also Helps in Numerical Stability
        self.alpha = nn.Parameter(torch.ones(features)) # nn.Parameter makes it leanable while training. alpha is multiplied
        self.bias = nn.Parameter(torch.zeros(features)) # Added

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim = -1, keepdim=True) # Usually mean() cancels the dim to which it is applied, but we want to keep it
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int=512, d_ff: int=2048, dropout: float=0.1) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2
    
    def forward(self, x):
        # linear_1: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        # linear_2: (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int=512, h: int=12, dropout: float=0.1) -> None:
        super().__init__()
        self.d_model = d_model # embedding dimension
        self.h = h # n_heads
        assert d_model % h == 0, "d_model must be divisible by h" # check beforehand
        self.d_k = d_model // h # Divide d_model into n_heads d_k's
        self.w_q = nn.Linear(d_model, d_model) # Query weight
        self.w_k = nn.Linear(d_model, d_model) # Key weight
        self.w_v = nn.Linear(d_model, d_model) # Value weight
        # Since, d_v (in paper) == d_k => d_v * h = d_model
        self.w_o = nn.Linear(d_model, d_model) # Output weight
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1) / math.sqrt(d_k))
        if mask is not None:
            # fill all masked with very large negative values, so that softmax makes them zero later
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Applying softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores # attention_scores will help in visualizing attentions

    def forward(self, q, k, v, mask): # mask -> used when some words do not want to interact with other words
        query = self.w_q(q) # (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float=0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        # return x + self.dropout(self.norm(sublayer(x))) # According to paper
        return x + self.dropout(sublayer(self.norm(x))) # But most implementations follow this
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        # For numerical stability, we apply log softmax
        # return torch.log_softmax(self.proj(x), dim=-1)
        return self.proj(x)


"""
    Datasets
"""
class BilingualDataset(Dataset):
    def __init__(self, src_data: list, tgt_data: list, src_lan: str, tgt_lan: str, tokenizer_src, tokenizer_tgt, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_lan = src_lan
        self.tgt_lan = tgt_lan
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx: int):
        src_text = self.src_data[idx]
        tgt_text = self.tgt_data[idx]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
        dec_input_tokens = dec_input_tokens[:self.seq_len - 2]
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # we subtract 2 for SOS and EOS tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # In decoder side, we only add SOS token

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence length is too long")
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0,
        )

        # Targeted output from the decoder
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) and (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    # torch.triu() gives every value above the diagonal
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0 # we want the opposite (everything above diagonal to be set as zero)

"""
    Configurations
"""

def get_config():
    return {
        "vocab_size": 8192,
        "batch_size": 16,
        "num_epochs": 10,
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
        "output_file": "bleu_scores_base_10.txt",
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

"""
    Helper Functions for Training
"""

def get_all_sentences(split, lang):
    with open(f"ted-talks-corpus/{split}.{lang}", 'r', encoding='utf-8') as file:
        sentences = [line.rstrip() for line in file]
        for sentence in sentences:
            yield sentence

def get_or_build_tokenizer(config, split, lang):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequence=2)
        tokenizer.train_from_iterator(get_all_sentences(split, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    # Build Tokenizers
    tokenizer_src = get_or_build_tokenizer(config, "train", "en")
    tokenizer_tgt = get_or_build_tokenizer(config, "train", "fr")

    with open("ted-talks-corpus/train.en", 'r', encoding='utf-8') as file:
        en_train = [line.rstrip() for line in file]

    with open("ted-talks-corpus/train.fr", 'r', encoding='utf-8') as file:
        fr_train = [line.rstrip() for line in file]

    with open("ted-talks-corpus/dev.en", 'r', encoding='utf-8') as file:
        en_dev = [line.rstrip() for line in file]

    with open("ted-talks-corpus/dev.fr", 'r', encoding='utf-8') as file:
        fr_dev = [line.rstrip() for line in file]

    with open("ted-talks-corpus/test.en", 'r', encoding='utf-8') as file:
        en_test = [line.rstrip() for line in file]

    with open("ted-talks-corpus/test.fr", 'r', encoding='utf-8') as file:
        fr_test = [line.rstrip() for line in file]

    train_dataset = BilingualDataset(en_train, fr_train, config["lang_src"], config["lang_tgt"], tokenizer_src, tokenizer_tgt, config["seq_len"])
    dev_dataset = BilingualDataset(en_dev, fr_dev, config["lang_src"], config["lang_tgt"], tokenizer_src, tokenizer_tgt, config["seq_len"])
    test_dataset = BilingualDataset(en_test, fr_test, config["lang_src"], config["lang_tgt"], tokenizer_src, tokenizer_tgt, config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0

    for sentence in en_train:
        indices = tokenizer_src.encode(sentence).ids
        max_len_src = max(len(indices), max_len_src)

    for sentence in fr_train:
        indices = tokenizer_tgt.encode(sentence).ids
        max_len_tgt = max(len(indices), max_len_tgt)

    print(f"Max length of source langauge: {max_len_src}")
    print(f"Max length of target langauge: {max_len_tgt}")

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dataloader, dev_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt

# Other methods such as taking random from top-k samples exists. But to keep the model simple, we take only the top most word greedily
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def generate_bleu_scores_file(model, test_dataloader, tokenizer_src, tokenizer_tgt, max_len, device, output_file, console_width=80):
    model.eval()
    bleu_metric = BLEUScore()

    with open(output_file, 'w', encoding='utf-8') as f:
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Generating BLEU scores"):
                encoder_input = batch['encoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)

                # Greedy decode
                model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

                source_text = batch['src_text'][0]
                target_text = batch['tgt_text'][0]
                model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

                # Calculate BLEU score for this pair
                bleu_score = bleu_metric([model_out_text], [[target_text]]).item()

                # Write to file in the specified format
                f.write('-' * console_width + '\n')
                f.write(f'SOURCE: {source_text}\n')
                f.write(f'TARGET: {target_text}\n')
                f.write(f'PREDICTED: {model_out_text}\n')
                f.write(f'BLEU Score: {bleu_score:.4f}\n')
                f.write('-' * console_width + '\n\n')  # Extra newline for spacing between entries

    print(f"BLEU scores and translations have been written to {output_file}")