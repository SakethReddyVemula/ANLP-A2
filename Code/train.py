import torch
import torch.nn as nn
import math
from utils import InputEmbeddings, PositionalEncoding, LayerNormalization, FeedForwardBlock, MultiHeadAttentionBlock, ResidualConnection, ProjectionLayer
from encoder import Encoder, EncoderBlock
from decoder import Decoder, DecoderBlock
from pathlib import Path
import wandb
import os
os.environ["WANDB_API_KEY"] = ""
from torchmetrics.text import BLEUScore
import torch.optim as optim
from tqdm import tqdm
from rouge_score import rouge_scorer
import sys

# Import from other files
from utils import BilingualDataset, causal_mask, get_config, get_weights_file_path, manage_saved_models
from utils import get_all_sentences, get_or_build_tokenizer, get_dataset, greedy_decode, generate_bleu_scores_file

class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # During inference, we only need the output of the encoder layer. Also we need to visualize the attention
    # Hence, we don't define forward methods, instead seperate encode, decode
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
"""
    Function for Building Transformer given Hyperparameters by combining all the blocks
"""
def build_transformer(
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int=512,
        N: int=6,
        h: int=8,
        dropout: float=0.1,
        d_ff: int=2048
) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoding blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoding blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) # Optimizes the code for faster convergence. Used in nn.Transformer also

    return transformer

def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size, tgt_vocab_size, config["seq_len"], config["seq_len"], config["d_model"], dropout=config["dropout"], N=config["N"], h=config["h"], d_ff=config["d_ff"])
    return model

def run_validation(model, validation_dataset, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    model.eval() # freeze the parameters

    count = 0

    source_texts = []
    expected = []
    expected_list = []
    predicted = []

    # size of the control window (just use a default value)
    console_width = 80

    # initialize the BLEU score
    bleu_metric = BLEUScore()

    # Initialize ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Autoregressive modeling
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            target_text_list = []
            target_text_list.append(target_text)
            source_texts.append(source_text)
            expected_list.append(target_text_list)
            expected.append(target_text)
            predicted.append(model_out_text)

            if count <= num_examples:
                # print to the console
                print_msg('-'*console_width)
                print_msg(f'SOURCE: {source_text}')
                print_msg(f'TARGET: {target_text}')
                print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                print_msg('-'*console_width)


    # Compute the BLEU metric
    bleu = bleu_metric(predicted, expected_list)
    
    # Compute ROUGE scores
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for ref, pred in zip(expected, predicted):
        scores = rouge_scorer_obj.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure

    # Average ROUGE scores
    for key in rouge_scores:
        rouge_scores[key] /= len(expected)

    # Log scores
    wandb.log({
        'validation/BLEU': bleu,
        'validation/ROUGE1': rouge_scores['rouge1'],
        'validation/ROUGE2': rouge_scores['rouge2'],
        'validation/ROUGEL': rouge_scores['rougeL'],
        'global_step': global_step
    })

    # Print scores
    print_msg(f"BLEU score: {bleu:.4f}")
    print_msg(f"ROUGE1 score: {rouge_scores['rouge1']:.4f}")
    print_msg(f"ROUGE2 score: {rouge_scores['rouge2']:.4f}")
    print_msg(f"ROUGEL score: {rouge_scores['rougeL']:.4f}")

    return bleu, rouge_scores


def train_model(config):
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure the folders are created
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, dev_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        del state

    # label smoothing helps in avoiding overfitting
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    wandb.define_metric("global_step")
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # shape: (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # shape: (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (batch, seq_len)

            # (batch, seq_len, tgt_vocab_size) -> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Log the loss
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})

            # Backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        if (epoch + 1) % 5 == 0:
            run_validation(model, dev_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
            
        manage_saved_models(config, epoch)

    generate_bleu_scores_file(model, test_dataloader, tokenizer_src, tokenizer_tgt, max_len=config["seq_len"], device=device, output_file=config["output_file"])


if __name__ == "__main__":
    config = get_config()
    
    wandb.init(
        # set the wandb project where this run will be logged
        project = "ANLP-A2-Transformers_from_scratch",
        config = config # keep track of all the hyperparameters and metadata
    )

    train_model(config)
