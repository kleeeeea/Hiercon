import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import batch_matrix_mul, element_wise_mul, zero_gating_softmax_normalize
torch.set_default_dtype(torch.float64)


class SentAttNet(nn.Module):
    def __init__(self, sent_feature_size=4):
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(sent_feature_size, sent_feature_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, sent_feature_size))
        self.context_weight = nn.Parameter(torch.Tensor(sent_feature_size, 1))
        self.context_bias = nn.Parameter(torch.Tensor(1))
        self._create_weights(mean=0.0, std=0.05)
        self.context_weight_history = []

    def _create_weights(self, mean=0.0, std=0.005):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(1, std)

        self.sent_bias.data.normal_(mean, std)

        self.context_bias.data.zero_()
        self.context_bias.requires_grad = False

    def forward(self, input):
        self.context_weight_history.append(
            self.context_weight.data.cpu().numpy().reshape(-1).copy())
        output = batch_matrix_mul(input, self.context_weight, self.context_bias)

        attn_score = zero_gating_softmax_normalize(output)

        return attn_score
