import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union
from .MiniMindBlock import MiniMindBlock
from .config import MiniMindConfig
from .RMSNorm import RMSNorm
from .RoPE import precompute_freqs_cis

class MiniMind(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        
        self.embed_token = nn.Embedding(args.vocab_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList(
            [MiniMindBlock(i, args) for i in args.num_hidden_layers]
        )
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        
        # RoPE预先计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=args.hidden_size // args.num_attention_heads, # RoPE作用在每个attention head上
            end=args.max_position_embeddings, # token 的最大数量
            rope_base=args.rope_theta,
            rope_scaling=args.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None,
        use_cache: bool=False,
        **kwargs
    ):
        bs, seq_len = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_value=None
        
        # 自回归推理
        past_key_values = past_key_values or [None]*len(self.layers)
        # kvcache的结构：past_key_values[layer] = (past_k, past_v)
        # past_k.shape: [bs, seq_len_cached, num_heads, head_dim]
        # past_key_values[0][0].shape[1]取得是 缓存的历史token数， 在同一次输入输出流程中所有层的缓存长度是一样的
        # seq_len_cached会随着forward次数的增加而增加
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )
        x = self.dropout(self.embed_token(input_ids))
        # 推理时seq_len恒为1，因为这里使用了kvcache，前面的token的k和v都被缓存了。训练时不使用kvcache
        # “历史长度”的增长体现在 start_pos 上，而不是 seq_len 上
        position_embeddings=(
            self.freqs_cos[start_pos:start_pos+seq_len],
            self.freqs_sin[start_pos:start_pos+seq_len],
        )
        presents = [] # 保存kvcache，每一个layer都有自己的kvcache
        # 依次经过每一层attention layer
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            x, present = layer(
                x,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        
        x = self.norm(x)
        return x, presents
    
class MiniMindForCasualLLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    
    def __init__(self, args: MiniMindConfig):
        super().__init__(args)
        self.model = MiniMind(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        # embed是将vocab_size映射到hidden_size,lm_head反之
        # 这里让它们权重共享
        self.model.embed_token.weight = self.lm_head.weight
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None,
        use_cache: bool=False,
        logits_to_keep: Union[int, torch.Tensor]=0,
        **args
    ):
        x, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # logits_to_keep 是整数，就保留最后n个位置；如果是张量，就是保留选定位置
        # 预测生成的时候只需要最后的logits来预测下一个token
        # 训练时保留全部
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        # logits 通常泛指 softmax 前的输出向量。
        # logits 是模型对每个候选词的原始打分，经过 softmax 后才变成概率。
        logits = self.lm_head(x[:, slice_indices, :])
        
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=x
        )
