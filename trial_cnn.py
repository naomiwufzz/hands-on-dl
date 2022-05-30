import torch
import torch.nn as nn
import torch.nn.functional as F
in_channel = 1  # 1个通道；
out_channel = 1  # 1个输出通道
kernel_size = 3  # 可以是标量可以是元组
batch_size = 1
bias = False
input_size = [batch_size, in_channel, 4, 4]  # 假设3个通道的图片，4*4大小
conv_layer = torch.nn.Conv2d(in_channel, out_channel, kernel_size, bias=bias)
input_feature_map = torch.randn(input_size)
output_feature_map = conv_layer(input_feature_map)
print(input_feature_map)
print(conv_layer.weight) # kernel 1*1*3*3 out_channels*in_channels*height*width，out_channels*in_channels是kernel数量
print(output_feature_map)

output_feature_map1 = F.conv2d(input_feature_map, conv_layer.weight)
print(output_feature_map1)


input = torch.randn(5, 5) # 卷积的输入特征图
kernel = torch.randn(3, 3) # 卷积核
bias = torch.randn(1) # 卷积偏置，默认输出通道数据等于1
# step1: 用原始的矩阵运算实现二维卷积
def matrix_multiplication_for_conv2d(input, kernel, stride=1, padding=0, bias =0):
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))

    # 获取大小
    input_h, input_w = input.shape
    kernel_h, kernel_w = kernel.shape
    output_w = (input_h - kernel_h) // stride + 1  # 卷积输出的宽度
    output_h = (input_w - kernel_w) // stride + 1 # 卷积输出的高度
    output = torch.zeros(output_h, output_w) # 初始化输出矩阵

    # 移动kernel
    for i in range(0, input_h-kernel_h+1, stride): # 对高度进行遍历
        for j in range(0, input_w-kernel_w+1, stride): # 对宽度进行遍历
            region = input[i:i+kernel_h, j:j+kernel_w] # 取出被卷积核滑动到的区域
            output[int(i/stride), int(j/stride)] = torch.sum(region * kernel) + bias  # 点乘，并赋值给输出位置的元素
    return output

# step2 用原始的矩阵运算实现二维卷积，先不考虑batchsize和channel，flatten版本
def matrix_multiplication_for_conv2d_flatten(input, kernel, stride=1, padding=0, bias =0):
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))

    # 获取大小
    input_h, input_w = input.shape
    kernel_h, kernel_w = kernel.shape
    output_w = (input_h - kernel_h) // stride + 1  # 卷积输出的宽度
    output_h = (input_w - kernel_w) // stride + 1 # 卷积输出的高度
    output = torch.zeros(output_h, output_w) # 初始化输出矩阵
    # 存储所有拉平后的特征区域
    region_matrix = torch.zeros(output.numel(), kernel.numel())  # numel给出张量的元素的总数
    kernel_matrix = kernel.reshape((kernel.numel(), 1)) # kernel的列向量形式
    # 移动kernel
    row_index = 0
    for i in range(0, input_h-kernel_h+1, stride): # 对高度进行遍历
        for j in range(0, input_w-kernel_w+1, stride): # 对宽度进行遍历
            region = input[i:i+kernel_h, j:j+kernel_w] # 取出被卷积核滑动到的区域
            region_vector = torch.flatten(region) # 把region拉平，形成一个一维张量
            region_matrix[row_index] = region_vector
            row_index += 1
            # output[int(i/stride), int(j/stride)] = torch.sum(region * kernel) + bias  # 点乘，并赋值给输出位置的元素
    output_matrix = region_matrix @ kernel_matrix
    output = output_matrix.reshape((output_h, output_w)) + bias
    return output

# step3: 用原始的矩阵运算实现二维卷积, 考虑batchsize维度和channel维度
def matrix_multiplication_for_conv2d_full(input, kernel, stride=1, padding=0, bias =0):
    # input， kernel都是4维的张量
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding, 0, 0, 0, 0)) # pad函数是从里到外的，反过来

    # 获取大小
    bs, in_channel, input_h, input_w = input.shape
    out_channel, in_channel, kernel_h, kernel_w = kernel.shape
    if bias is None: # bias是加在输出通道上的
        bias = torch.zeros(out_channel)

    output_w = (input_h - kernel_h) // stride + 1  # 卷积输出的宽度
    output_h = (input_w - kernel_w) // stride + 1 # 卷积输出的高度
    output = torch.zeros(bs, out_channel, output_h, output_w) # 初始化输出矩阵

    # 移动kernel
    for ind in range(bs):
        for oc in range(out_channel): # input加和
            for ic in range(in_channel):
                # 对二维的卷积计算结果
                for i in range(0, input_h-kernel_h+1, stride): # 对高度进行遍历
                    for j in range(0, input_w-kernel_w+1, stride): # 对宽度进行遍历
                        region = input[ind, ic, i:i+kernel_h, j:j+kernel_w] # 取出被卷积核滑动到的区域
                        output[ind, oc, int(i/stride), int(j/stride)] += torch.sum(region * kernel[oc, ic])  # 点乘，并赋值给输出位置的元素
            output[ind, oc] += bias[oc]

    return output

# 矩阵运算实现卷积的结果
mat_mul_conv_output = matrix_multiplication_for_conv2d(input, kernel, padding=1, bias=bias, stride=2)
print(mat_mul_conv_output)

# 调用torch api卷积的结果
pytorch_api_conv_output = F.conv2d(input.reshape((1, 1, input.shape[0], input.shape[1])), kernel.reshape((1, 1, kernel.shape[0], kernel.shape[1])), padding=1, bias=bias, stride=2).squeeze(0).squeeze(0)
print(pytorch_api_conv_output)

# pytorch_api_conv_output = F.conv2d(input.reshape((1, 1, input.shape[0], input.shape[1])), kernel.reshape((1, 1, kernel.shape[0], kernel.shape[1])), padding=1, bias=bias)

flag1 = torch.allclose(mat_mul_conv_output, pytorch_api_conv_output) # 判断是否足够接近
print(flag1)
# 矩阵运算实现卷积的结果，flatten input的版本
mat_mul_conv_output_flatten = matrix_multiplication_for_conv2d_flatten(input, kernel, padding=1, bias=bias, stride=2)
# 验证flatten方式的结果和官方结果
flag2 = torch.allclose(mat_mul_conv_output_flatten, pytorch_api_conv_output)
print(flag2)

# 验证第三种
input = torch.randn(2, 2, 5, 5)  # bs in_channel, in_h, in_w,
kernel = torch.randn(3, 2, 3, 3)  # out_channel, in_channel, kernel_h, kernel_w
bias = torch.randn(3) # 和out_channel 一致
# 验证matrix_multiplication_for_conv2d_full和torch官方api结果一致性
pytorch_conv2d_api_output = F.conv2d(input, kernel, bias=bias, padding=1, stride=2)
mm_conv2d_full_output = matrix_multiplication_for_conv2d_full(input, kernel, padding=1, bias=bias, stride=2)
flag3 = torch.allclose(pytorch_conv2d_api_output, mm_conv2d_full_output)
# print(pytorch_conv2d_api_output)
# print(mm_conv2d_full_output)
print(flag3)