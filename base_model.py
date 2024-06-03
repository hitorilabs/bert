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


class BertModel(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
    
    def forward(self, input_ids, attention_mask=None):
        encoder_output = self.encoder(
            self.embeddings(input_ids),
            src_key_padding_mask=(attention_mask==0),
        )
        return encoder_output


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_dim)
        
        self.layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        
    def forward(self, input_ids):
        position_ids = self.position_ids[:, 0:input_ids.shape[1]]
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.layer_norm(embeddings)
        
        return embeddings



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
            x = mod(x, src_key_padding_mask=src_key_padding_mask)
        return x


class BertEncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            bias=True,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps, bias=True, elementwise_affine=True)

        self.ffn = BertFeedForwardLayer(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps, bias=True, elementwise_affine=True)

    def forward(self, x, src_key_padding_mask=None):
        x = self.self_attn_norm(x + self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False)[0])
        x = self.ffn_norm(x + self.ffn(x))
        return x


class BertFeedForwardLayer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_dim, config.feedforward_dim, bias=True)
        self.intermediate_act_fn = nn.GELU()
        self.output_dense = nn.Linear(config.feedforward_dim, config.hidden_dim, bias=True)
    
    def forward(self, x):
        x = self.intermediate_dense(x)
        x = self.intermediate_act_fn(x)
        x = self.output_dense(x)
        return x
    
