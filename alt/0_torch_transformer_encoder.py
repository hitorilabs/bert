import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field

@dataclass
class BertConfig:
    hidden_dim: int = 1024
    feedforward_dim: int = field(init=False)

    num_layers: int = 24
    num_heads: int = 16
    dropout: float = 0.1
    activation = F.gelu
    layer_norm_eps: float = 1e-12
    dtype = torch.float32

    # embedding config
    vocab_size: int = 30522
    max_position_embeddings = 512 # Max sequence length

    def __post_init__(self):
        self.feedforward_dim = self.hidden_dim * 4


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, layer_norm_eps=1e-12, dropout_prob=0.1):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False
        )
        
    def forward(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, 0:seq_length]
        
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = word_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertModel(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_dim,
            max_position_embeddings=config.max_position_embeddings,
        )
        
        # note: tokenizer output is batch first, this was causing issues loading
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation=config.activation,
                layer_norm_eps=config.layer_norm_eps,
                batch_first=True,
                norm_first=False,
                bias=True,
                dtype=config.dtype,
            ),
            num_layers=config.num_layers,
            norm=None,
            enable_nested_tensor=True,
            mask_check=True,
        )
    
    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embeddings(input_ids)
        encoder_output = self.encoder(
            embeddings,
            src_key_padding_mask=(attention_mask==0),
        )
        return encoder_output
