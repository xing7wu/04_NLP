"""
attention
"""
import torch.nn as nn
import torch


# 自定义Attention类
class Attention(nn.Module):
    # 前向传播
    def forward(self, decoder_hiddens, encoder_outputs):
        # 1. 相关性计算
        attn_scores = torch.bmm(decoder_hiddens, encoder_outputs.mT)
        # 2. 注意力权重计算
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # 3. 上下文向量计算
        context_vectors = torch.bmm(attn_weights, encoder_outputs)
        # 4. 解码信息融合
        combined_vectors = torch.cat([decoder_hiddens, context_vectors], dim=-1)

        return combined_vectors


if __name__ == '__main__':
    attn = Attention()

    # 定义参数
    N = 32
    L_enc = 12
    L_dec = 15
    hidden_size = 128

    # 定义解码器隐藏状态、编码器的输出
    decoder_hiddens = torch.randn(N, L_dec, hidden_size)
    encoder_outputs = torch.randn(N, L_enc, hidden_size)

    combined_vectors = attn(decoder_hiddens, encoder_outputs)
    print(combined_vectors.shape)
