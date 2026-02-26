import torch
import torch.nn as nn
from .config import MiniMindConfig
from transformers.activations import ACT2FN

class FeedForward(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size*8/3)
            args.intermediate_size = 64*((intermediate_size+64-1)//64)
            
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]
    
    def forward(self, x: torch.Tensor):
        return self.dropout(self.down_proj(self.up_proj(x)*self.act_fn(self.gate_proj(x))))