import torch.nn as nn
from utils import InputEmbeddings, PositionalEncoding, LayerNormalization, FeedForwardBlock, MultiHeadAttentionBlock, ResidualConnection, ProjectionLayer

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # q, k, v all = x in encoder block => self attention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # Multi-head Attention step
        x = self.residual_connections[1](x, self.feed_forward_block) # Feed Forward step
        return x
    
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features) # Layer Normalization at the end of all the blocks

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)