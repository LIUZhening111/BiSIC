import torch
from torch import nn
from compressai.layers import ResidualBlock
import torch.nn.functional as F


class EfficientScaledDotProductAttention(nn.Module):
    # b*head, C', h*w  as input
    def forward(self, query, key, value, mask=None):
        n = query.size()[-1]
        # scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        # context = key.matmul(value.transpose(-2, -1)) / math.sqrt(n)
        context = key.matmul(value.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = (context.transpose(-2, -1)).matmul(query)
        return attention

class EfficientMultiHeadAttention(nn.Module):

    def __init__(self, in_features, embed_QK, embed_V, head_num, bias=True, activation=None):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(EfficientMultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.embed_QK = embed_QK
        self.embed_V = embed_V
        self.conv_q = nn.Conv2d(in_features, embed_QK, 1)
        self.conv_k = nn.Conv2d(in_features, embed_QK, 1)
        self.conv_v = nn.Conv2d(in_features, embed_V, 1)
        self.projection_o = nn.Conv2d(embed_V, in_features, bias)

    def forward(self, q, k, v, mask=None):
        assert q.size() == k.size() == v.size()
        b, _, h, w = q.size()
        q, k, v = self.conv_q(q), self.conv_k(k), self.conv_v(v)
        q = q.reshape(b, self.embed_QK, h*w).permute(0, 2, 1)
        k = k.reshape(b, self.embed_QK, h*w).permute(0, 2, 1)
        v = v.reshape(b, self.embed_V, h*w).permute(0, 2, 1)  # b, h*w, embedC
        
        q = F.softmax(self._reshape_to_batches(q).permute(0, 2, 1), dim=1) # b*head, C', h*w
        k = F.softmax(self._reshape_to_batches(k).permute(0, 2, 1), dim=2)
        v = self._reshape_to_batches(v).permute(0, 2, 1)
        y = (EfficientScaledDotProductAttention()(q, k, v, mask)).permute(0, 2, 1)   # b*head, C', h*w  --> b*head, h*w, C'
        y = self._reshape_from_batches(y)   # b, h*w, C
        y = y.permute(0, 2, 1).reshape(b, self.embed_V, h, w)
        y = self.projection_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim).permute(0, 2, 1, 3).reshape(batch_size * self.head_num, seq_len, sub_dim) # b*head, h*w, C'

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature).permute(0, 2, 1, 3).reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )



class MutualAttention_master(nn.Module):
    def __init__(self, channels):
        super(MutualAttention_master, self).__init__()
        self.rb1 = ResidualBlock(channels, channels)
        self.rb2 = ResidualBlock(channels, channels)
        self.cross_attnA = EfficientMultiHeadAttention(in_features=channels, embed_QK=channels//2, embed_V=channels//2, head_num=8, bias=True, activation=None)
        self.self_attenA = EfficientMultiHeadAttention(in_features=channels, embed_QK=channels//2, embed_V=channels//2, head_num=8, bias=True, activation=None)
        self.cross_attenB = EfficientMultiHeadAttention(in_features=channels, embed_QK=channels//2, embed_V=channels//2, head_num=8, bias=True, activation=None)
        self.self_attenB = EfficientMultiHeadAttention(in_features=channels, embed_QK=channels//2, embed_V=channels//2, head_num=8, bias=True, activation=None)

        self.refine = nn.Sequential(
            ResidualBlock(channels*3, channels*2),
            ResidualBlock(channels*2, channels))

    def forward(self, x_left, x_right):
        B, C, H, W = x_left.size()
        identity_left, identity_right = x_left, x_right
        x_left, x_right = self.rb2(self.rb1(x_left)), self.rb2(self.rb1(x_right))
        A_right_to_left, A_left_to_right = self.cross_attnA(q=x_left, k=x_right, v=x_right), self.cross_attnA(q=x_right, k=x_left, v=x_left)
        S_leftA, S_rightA = self.self_attenA(q=A_right_to_left, k=A_right_to_left, v=A_right_to_left), self.self_attenA(q=A_left_to_right, k=A_left_to_right, v=A_left_to_right)
        B_right_to_left, B_left_to_right = self.cross_attenB(q=x_right, k=x_right, v=x_left), self.cross_attenB(q=x_left, k=x_left, v=x_right) 
        S_leftB, S_rightB = self.self_attenB(q=B_right_to_left, k=B_right_to_left, v=B_right_to_left), self.self_attenB(q=B_left_to_right, k=B_left_to_right, v=B_left_to_right)
        compact_left = identity_left + self.refine(torch.cat((S_leftA, S_leftB, x_left), dim=1))
        compact_right = identity_right + self.refine(torch.cat((S_rightA, S_rightB, x_right), dim=1))
        return compact_left, compact_right
