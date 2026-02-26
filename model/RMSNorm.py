import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # 初始化为全1的权重
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size
    
    def _norm(self, x: torch.Tensor):
        div = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x * div
    
    def forward(self, x: torch.Tensor):
        # 将x变为float32提高计算精度，后续再用type_as变换会原先类型
        return self.weight * self._norm(x.float()).type_as(x)