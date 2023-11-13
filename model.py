import torch
from torch import nn


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_head, dropout_rate=0.2):
        super().__init__()
        self.d_model = d_model
        self.queries = nn.Linear(d_model, d_head, bias=False)
        self.keys = nn.Linear(d_model, d_head, bias=False)
        self.values = nn.Linear(d_model, d_head, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        Q = self.queries(X)
        K = self.keys(X)
        V = self.values(X)
        A = Q @ K.T(-2, -1)
        A = A / torch.sqrt(self.d_model)
        A = torch.softmax(A, dim=-1)
        A = self.dropout(A)
        new_V = A @ V
        return new_V


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout_rate=0.2):
        super().__init__()
        d_head = d_model // num_heads
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(AttentionHead(d_model, d_head))
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        heads_outputs = []
        for head in self.heads:
            heads_outputs.append(head(X))
        outputs_concated = torch.cat(heads_outputs, dim=-1)
        output = self.linear(outputs_concated)
        output = self.dropout(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, dropout_rate=0.2):
        super().__init__()
        self.mh_attention_layer = MultiHeadAttention(num_heads, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ff_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_rate),
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, X):
        attention_output = self.mh_attention_layer(X)
        attention_output = attention_output + X  # residual connection
        attention_output = self.layer_norm1(attention_output)
        ff_output = self.ff_layer(attention_output)
        ff_output = ff_output + X  # residual connection
        ff_output = self.layer_norm2(ff_output)
        return ff_output


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_window,
        d_model=512,
        d_ff=2048,
        heads_per_block=8,
        num_layers=6,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(context_window, d_model)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(heads_per_block, d_model, d_ff))
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, X):
        tok_emb = self.token_embedding(X)
        pos_emb = self.pos_embedding(torch.arange(X.shape[1], device=self.device))
        X = tok_emb + pos_emb
        X = self.blocks(X)
        logits = self.linear(X)
        return logits
