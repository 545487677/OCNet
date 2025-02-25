from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.modules import LayerNorm, SelfMultiheadAttention
from .triangular_multiplicative_update import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming

class TransformerEncoderTriLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)

        self.tri_out = TriangleMultiplicationOutgoing(c_z=attention_heads, c_hidden=attention_heads*2)
        self.tri_in = TriangleMultiplicationIncoming(c_z=attention_heads, c_hidden=attention_heads*2)
    
        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln


    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool=False,
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        sz = x.size()
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        # new added
        if padding_mask is not None:
            attn_mask = torch.ones(sz[0], sz[1], sz[1]).type_as(x)
            attn_mask.masked_fill_(
                padding_mask.unsqueeze(1).repeat(1, sz[1], 1).to(torch.bool),
                0,
            )
            attn_mask.masked_fill_(
                padding_mask.unsqueeze(-1).repeat(1, 1, sz[1]).to(torch.bool),
                0,
            )
        else:
            attn_mask = None
        if attn_bias is not None:
            attn_bias[attn_bias == float("-inf")] = 0.0
            attn_bias = attn_bias.view(sz[0], -1, sz[1], sz[1]).permute(0, 2, 3, 1).contiguous()
            attn_bias = self.tri_in(attn_bias, mask=attn_mask) + attn_bias
            attn_bias = self.tri_out(attn_bias, mask=attn_mask) + attn_bias
            attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous()
            attn_bias = attn_bias.view(-1, sz[1], sz[1])
            
        x = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs

class TransformerEncoderWithTriPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None
        
        if not no_final_head_layer_norm:
            self.final_head_layer_norm = LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None

        self.layers = nn.ModuleList(
            [
                TransformerEncoderTriLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                    
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self,
        emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        bsz = emb.size(0)
        seq_len = emb.size(1)        
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask
        mask_pos = padding_mask

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        for i in range(len(self.layers)):
            x, attn_mask, _ = self.layers[i](x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True)

        
        mask_pos_t = torch.ones((x.size(0),x.size(1)),device = x.device)
        mask_pos_t = mask_pos_t.type_as(x)
        if mask_pos is not None:
            mask_pos_t.masked_fill_(
                mask_pos.to(torch.bool),
                0,
            )
        x_norm = (x.float().norm(dim=-1) - math.sqrt(x.size(-1))).abs()
        mask_pos_t.masked_fill_(
            x_norm <= 1,
            0,
        )
        mask_pos_t = mask_pos_t.to(torch.bool)
        if mask_pos_t.any():
            x_norm = x_norm[mask_pos_t].mean()
        else:
            x_norm = torch.zeros(1,device=mask_pos_t.device)
            
        
        if self.final_layer_norm !=  None:
            x = self.final_layer_norm(x)
        
        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        delta_pair_repr = delta_pair_repr.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        delta_pair_repr_norm = delta_pair_repr.float().norm(dim=-1)
        mask_pos_t = torch.ones_like(delta_pair_repr_norm)
        if mask_pos is not None:
            mask_pos_t.masked_fill_(
                mask_pos.unsqueeze(1).to(torch.bool),
                0,
            )
            mask_pos_t = mask_pos_t.permute(0, 2, 1)
            mask_pos_t.masked_fill_(
                mask_pos.unsqueeze(1).to(torch.bool),
                0,
            )
            mask_pos_t = mask_pos_t.permute(0, 2, 1)

        delta_pair_repr_norm = (delta_pair_repr_norm - math.sqrt(delta_pair_repr.size(-1))).abs()
        mask_pos_t.masked_fill_(
            delta_pair_repr_norm <= 1,
            0,
        )
        mask_pos_t = mask_pos_t.to(torch.bool)
        if mask_pos_t.any():
            delta_pair_repr_norm = delta_pair_repr_norm[mask_pos_t].mean()
        else:
            delta_pair_repr_norm = torch.zeros(1,device=mask_pos_t.device)
        if self.final_head_layer_norm !=None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm