from torch import nn
import torch
import torch.nn.functional as F
from modelss.DSFA_SwinNet.DSFA import DSFA as skipblock


# 定义一个包含空洞卷积、批量归一化和ReLU激活函数的子模块
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            # 空洞卷积，通过调整dilation参数来捕获不同尺度的信息
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),  # 批量归一化
            nn.ReLU()  # ReLU激活函数
        ]
        super(ASPPConv, self).__init__(*modules)


# 定义一个全局平均池化后接卷积、批量归一化和ReLU的子模块
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1x1卷积
            nn.BatchNorm2d(out_channels),  # 批量归一化
            nn.ReLU())  # ReLU激活函数

    def forward(self, x):
        size = x.shape[-2:]  # 保存输入特征图的空间维度
        x = super(ASPPPooling, self).forward(x)
        # 通过双线性插值将特征图大小调整回原始输入大小
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class PAR(nn.Module):
    def __init__(self, in_channels, resolution=16, atrous_rates=[2, 4, 8]):
        super().__init__()
        out_channels = 256  # 输出通道数

        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1x1卷积用于降维
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 根据不同的膨胀率添加空洞卷积模块
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 添加全局平均池化模块
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        out_num = len(modules)
        # print("out_num : ", out_num)

        # 将所有模块的输出融合后的投影层
        self.project = nn.Sequential(
            nn.Conv2d(out_num * out_channels, out_channels, 1, bias=False),  # 融合特征后降维
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))  # 防止过拟合的Dropout层

        self.atn = skipblock(out_channels, resolution, resolution)

    def forward(self, x):
        res = []
        # 对每个模块的输出进行收集
        for conv in self.convs:
            res.append(conv(x))
        # 将收集到的特征在通道维度上拼接
        res = torch.cat(res, dim=1)
        # 对拼接后的特征进行处理
        return self.atn(self.project(res))