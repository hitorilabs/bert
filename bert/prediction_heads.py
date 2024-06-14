import torch
import torch.nn as nn


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, embeddings_weight):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # tie weights and bias
        self.decoder.weight = embeddings_weight
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.transform_act_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.transform_act_fn(x)
        x = self.layer_norm(x)
        return x
