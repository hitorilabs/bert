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
    layer_norm_eps: float = 1e-12
    dtype = torch.float32

    vocab_size: int = 30522
    max_position_embeddings = 512

    def __post_init__(self):
        self.feedforward_dim = self.hidden_dim * 4


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, layer_norm_eps=1e-12, dropout_prob=0.1):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
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
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertFeedForwardLayer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_dim, config.feedforward_dim, bias=True)
        self.intermediate_act_fn = nn.GELU()
        self.intermediate_dropout = nn.Dropout(config.dropout)
        self.output_dense = nn.Linear(config.feedforward_dim, config.hidden_dim, bias=True)
        self.output_dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.intermediate_dense(x)
        x = self.intermediate_act_fn(x)
        x = self.intermediate_dropout(x)
        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x
    
class BertEncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            bias=True,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps, bias=True, elementwise_affine=True)
        self.self_attn_dropout = nn.Dropout(config.dropout)

        self.ffn = BertFeedForwardLayer(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps, bias=True, elementwise_affine=True)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        x = self.self_attn_norm(x + self.self_attn_dropout(self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]))
        x = self.ffn_norm(x + self.ffn(x))
        return x


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertEncoderLayer(config) for _ in range(config.num_layers)])
    
    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="mask",
            target_type=x.dtype
        )

        for mod in self.layer:
            x = mod(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return x

class BertModel(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_dim,
            max_position_embeddings=config.max_position_embeddings,
        )

        self.encoder = BertEncoder(config)
    
    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embeddings(input_ids)
        encoder_output = self.encoder(
            embeddings,
            src_key_padding_mask=(attention_mask==0),
        )
        return encoder_output

