import torch
import torch.nn as nn
from .config import MiniMindConfig
from .GQA import Attention
from .FFN import FeedForward
from .RMSNorm import RMSNorm

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, args: MiniMindConfig):
        super().__init__()
        self.layer_id = layer_id
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(args)
        self.attn_rms_norm = RMSNorm(self.hidden_size, args.rms_norm_eps)
        self.self_ffn = FeedForward(args)
        self.ffn_rms_norm = RMSNorm(self.hidden_size, args.rms_norm_eps)
        
    def forward(self, x: torch.Tensor, position_embedding, past_ket_value=None, use_cache=False, attention_mask=None):
        residual = x
        x, present_key_value = self.self_attn(
            self.attn_rms_norm(x),
            position_embedding,
            past_ket_value,
            use_cache,
            attention_mask
        )
        x = residual + x
        x = x + self.self_ffn(self.ffn_rms_norm(x))
        return x, present_key_value