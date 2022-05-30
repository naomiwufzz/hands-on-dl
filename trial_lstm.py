import torch.nn as nn
import torch

# torch.nn.LSTM
# 实现 LSTM 和 LSTMP 源码
# 定义常量
bs, T, i_size, h_size = 2, 3, 4, 5  # batch size，长度，input feature(embedding_dim), hidden
proj_size = 3
input = torch.randn(bs, T, i_size)
c0 = torch.randn(bs, h_size)  # 初始值，并不训练
h0 = torch.randn(bs, h_size)

# 调用官方LSTM API，实例化
# proj_size是把输出压缩，并不压缩记忆单元，是优化参数的方式
lstm_layer = nn.LSTM(i_size, h_size, batch_first=True, proj_size =proj_size) # 高版本lstm才有的
# 根据官方接口要求h0应该是三维的，在第0维扩一维（因为现在是一层），就是1*bs*h_size
output, (h_hinal, c_final) = lstm_layer(input, (h0.unsqueeze(0), c0.unsqueeze(0)))
# 看一下参数
for k, v in lstm_layer.named_parameters():
    print(k, v.shape)


# 自己写一个lstm模型
def lstm_forward(input, initial_state, w_ih, w_hh, b_ih, b_hh, w_hr=None):
    h0, c0 = initial_state
    bs, T, i_size = input.shape
    h_size = w_ih.shape[0] // 4

    prev_h = h0
    prev_c = c0
    batch_w_ih = w_ih.unsqueeze(0).repeat(bs, 1, 1)  # [4 * h_size, i_size],要扩出batch的维度，并复制batch个
    batch_w_hh = w_hh.unsqueeze(0).repeat(bs, 1, 1)  # [4 * h_size, h_size]
    # 处理降维
    if w_hr is not None: # 配置了proj
        p_size = w_hr.shape[0]
        output_size = p_size
        batch_w_hr = w_hr.unsqueeze(0).repeat(bs, 1, 1)
    else:
        output_size = h_size
    output_size = h_size # 有proj的话output size不一定等于hidden size
    output = torch.zeros(bs, T, output_size)  # 初始化一个输出序列
    for t in range(T):
        x = input[:, t, :]  # 当前时刻的输入向量 bs*i_size
        w_times_x = torch.bmm(batch_w_ih, x.unsqueeze(-1))  # [bs, 4*h_size, 1]
        w_times_x = w_times_x.squeeze(-1)  # [bs, 4*h_size]

        w_times_h_prev = torch.bmm(batch_w_hh, prev_h.unsqueeze(-1))  # [bs, 4*h_size, 1]
        w_times_h_prev = w_times_h_prev.squeeze(-1)

        # 分别计算输入们（i），遗忘门（f），cell门（g），输出门（o）
        i_t = torch.sigmoid(w_times_x[:, :h_size] + w_times_h_prev[:, :h_size] + b_ih[:h_size] + b_hh[:h_size])
        f_t = torch.sigmoid(
            w_times_x[:, h_size:2 * h_size] + w_times_h_prev[:, h_size:2 * h_size] + b_ih[h_size:2 * h_size] + b_hh[
                                                                                                         h_size:2 * h_size])
        g_t = torch.tanh(
            w_times_x[:,h_size * 2:3 * h_size] + w_times_h_prev[:,h_size * 2:3 * h_size] + b_ih[
                                                                                       h_size * 2:3 * h_size] + b_hh[
                                                                                                                h_size * 2:3 * h_size])
        o_t = torch.sigmoid(
            w_times_x[:,h_size * 3:] + w_times_h_prev[:,h_size * 3:] + b_ih[h_size * 3:] + b_hh[h_size * 3:])
        prev_c = f_t * prev_c + i_t * g_t
        prev_h = o_t * torch.tanh(prev_c) # [bs*h_size]

        if w_hr is not None:
            prev_h = torch.bmm(batch_w_hr, prev_h.unsqueeze(-1)) # [bs, p_size, 1]
            prev_h = prev_h.squeeze(-1) # [bs, p_size]

        output[:, t, :] = prev_h # 更新输出
    return output, (prev_h, prev_c)

output_custom, (h_hinal_custom, c_final_custom)  = lstm_forward(input, (h0, c0), lstm_layer.weight_ih_l0, lstm_layer.weight_hh_l0, lstm_layer.bias_ih_l0, lstm_layer.bias_hh_l0)