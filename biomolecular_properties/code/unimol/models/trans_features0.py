import logging

import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from .unimol import UniMolModel, base_architecture, NonLinearHead
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from .transformer_encoder_with_tri_pair import TransformerEncoderWithTriPair


logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS]) #torch.Size([16, 512])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@register_model("trans_features0")
class TransMixFeatures0Model(BaseUnicoreModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--mol-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--recycling",
            type=int,
            default=1,
            help="recycling nums of decoder",
        )
        parser.add_argument(
            "--only-rmsd",
            action="store_true",
            help="only predict rmsd",
        )
        parser.add_argument(
            "--output-type",
            type=str,
            help="output type",
        )
        parser.add_argument(
            "--mol-encoder-layers",
            type=int,
            default=15,
            help="recycling nums of decoder",
        )
        parser.add_argument(
            "--pocket-encoder-layers",
            type=int,
            default=15,
            help="recycling nums of decoder",
        )


    def __init__(self, args, mol_dictionary, pocket_dictionary):
        super().__init__()
        base_model_architecture(args)

        self.args = args
        self.mol_dictionary = mol_dictionary
        self.mol_model = UniMolModel(args.mol, mol_dictionary)

        K = 128
        dict_size = len(mol_dictionary) + len(mol_dictionary)
        n_edge_type = dict_size * dict_size
        self.concat_gbf = GaussianLayer(K, n_edge_type)

        self.cm_gbf = GaussianLayer(K, n_edge_type)

        self.classification_heads = ClassificationHead(
            input_dim=self.args.mol.encoder_embed_dim,
            inner_dim=self.args.mol.encoder_embed_dim // 2, 
            num_classes=self.args.mol.encoder_embed_dim,
            activation_fn=self.args.mol.pooler_activation_fn,
            pooler_dropout=self.args.mol.pooler_dropout,
        )
        self.feature_head = NonLinearHead(
            input_dim=8,
            out_dim=self.args.mol.encoder_embed_dim,
            activation_fn='relu',
        )
        self.final_head = NonLinearHead(
            input_dim=2 * self.args.mol.encoder_embed_dim,
            out_dim=self.args.num_classes,
            activation_fn='relu',
        )
        

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.dictionary)

    def forward(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_coord,
        mol_src_edge_type,
        mol_feature,
        **kwargs
    ):



        def get_dist_features(dist, et, flag):
            if flag == 'mol':
                n_node = dist.size(-1)
                gbf_feature = self.mol_model.gbf(dist, et)
                gbf_result = self.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            elif flag == 'pocket':
                n_node = dist.size(-1)
                gbf_feature = self.pocket_model.gbf(dist, et)
                gbf_result = self.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            elif flag == 'concat':
                n_node = dist.size(-1)
                gbf_feature = self.concat_gbf(dist, et)
                gbf_result = self.concat_gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            else:
                return None
    
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = get_dist_features(mol_src_distance, mol_src_edge_type, 'mol')
        mol_outputs = self.mol_model.encoder(mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias)
        mol_encoder_rep = mol_outputs[0]
        logits = self.classification_heads(mol_encoder_rep)

        feature_logits = self.feature_head(mol_feature.type_as(self.feature_head.linear1.weight))


        logits = torch.cat([logits, feature_logits], dim=-1)
        logits = self.final_head(logits.type_as(self.final_head.linear1.weight))


        

        return logits
    

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

class NonLinearHead(nn.Module):
    """Head for simple regression tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x=self.linear1(x)
        x=self.activation_fn(x)
        x=self.linear2(x)
        return x

class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

@register_model_architecture("trans_features0", "trans_features0")
def base_model_architecture(args):

    parser = argparse.ArgumentParser()
    args.mol = parser.parse_args([])
    args.pocket = parser.parse_args([])

    args.mol.encoder_layers = getattr(args, "mol_encoder_layers", 15) # 15
    args.mol.encoder_embed_dim = getattr(args, "mol_encoder_embed_dim", 512)
    args.mol.encoder_ffn_embed_dim = getattr(args, "mol_encoder_ffn_embed_dim", 2048)
    args.mol.encoder_attention_heads = getattr(args, "mol_encoder_attention_heads", 64)
    args.mol.dropout = getattr(args, "mol_dropout", 0.1)
    args.mol.emb_dropout = getattr(args, "mol_emb_dropout", 0.1)
    args.mol.attention_dropout = getattr(args, "mol_attention_dropout", 0.1)
    args.mol.activation_dropout = getattr(args, "mol_activation_dropout", 0.0)
    args.mol.pooler_dropout = getattr(args, "mol_pooler_dropout", 0.0)
    args.mol.max_seq_len = getattr(args, "mol_max_seq_len", 512)
    args.mol.activation_fn = getattr(args, "mol_activation_fn", "gelu")
    args.mol.pooler_activation_fn = getattr(args, "mol_pooler_activation_fn", "tanh")
    args.mol.post_ln = getattr(args, "mol_post_ln", False)
    args.mol.contrastive_global_negative = getattr(args, "mol_contrastive_global_negative", False)
    args.mol.masked_token_loss = -1.0
    args.mol.masked_coord_loss = -1.0
    args.mol.masked_dist_loss = -1.0
    args.mol.masked_bond_loss = -1.0
    args.mol.contrastive_loss = -1.0
    args.mol.fingerprint_loss = -1.0
    args.mol.x_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0
    args.mol.masked_coord_dist_loss = -1.0

    args.pocket.encoder_layers = getattr(args, "pocket_encoder_layers", 15) # 15
    args.pocket.encoder_embed_dim = getattr(args, "pocket_encoder_embed_dim", 512)
    args.pocket.encoder_ffn_embed_dim = getattr(args, "pocket_encoder_ffn_embed_dim", 2048)
    args.pocket.encoder_attention_heads = getattr(args, "pocket_encoder_attention_heads", 64)
    args.pocket.dropout = getattr(args, "pocket_dropout", 0.1)
    args.pocket.emb_dropout = getattr(args, "pocket_emb_dropout", 0.1)
    args.pocket.attention_dropout = getattr(args, "pocket_attention_dropout", 0.1)
    args.pocket.activation_dropout = getattr(args, "pocket_activation_dropout", 0.0)
    args.pocket.pooler_dropout = getattr(args, "pocket_pooler_dropout", 0.0)
    args.pocket.max_seq_len = getattr(args, "pocket_max_seq_len", 512)
    args.pocket.activation_fn = getattr(args, "pocket_activation_fn", "gelu")
    args.pocket.pooler_activation_fn = getattr(args, "pocket_pooler_activation_fn", "tanh")
    args.pocket.post_ln = getattr(args, "pocket_post_ln", False)
    args.pocket.contrastive_global_negative = getattr(args, "pocket_contrastive_global_negative", False)
    args.pocket.masked_token_loss = -1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.masked_bond_loss = -1.0
    args.pocket.contrastive_loss = -1.0
    args.pocket.fingerprint_loss = -1.0
    args.pocket.x_norm_loss = -1.0
    args.pocket.delta_pair_repr_norm_loss = -1.0
    args.pocket.masked_coord_dist_loss = -1.0
    
    base_architecture(args)