import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

from modelss.DSFA_SwinNet.DSFA import DSFA as skipblock
from modelss.DSFA_SwinNet.PAR import PAR as bottleneckblock


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    if len(x.shape) == 4:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    elif len(x.shape) == 5:
        _, B, H, W, C = x.shape

        x = x.view(3, B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(3, -1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)  # 进行下采样
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)  # 展平
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim,
                                   bias=False)  # 将原始特征图切割为4倍dim，再拼接输出2倍dim。因此输入channel = C，输出 chanel = 2C
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # [B, H, W, C]
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # 切割为4C
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)

        # 将4C变为2C
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock_Adaptive(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, window_size_list=[7, 14], shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_size_list = window_size_list  # [W1,...Wn]
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.win_cnt = len(window_size_list)
        self.n_group = self.win_cnt
        self.channel = self.dim // self.n_group
        assert self.dim == self.channel * self.n_group
        self.gnum_heads = num_heads // self.win_cnt
        # assert num_heads == self.gnum_heads * self.win_cnt
        self.gchannel = self.channel // self.gnum_heads
        assert self.channel == self.gchannel * self.gnum_heads

        self.Swin_block = nn.ModuleList([
            SwinTransformerBlock(
                dim=self.channel,
                num_heads= self.gnum_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,  # block必须成对使用
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i, window_size in enumerate(window_size_list)])

    def forward(self, x, masks=None):
        # B, L, C = x.shape  # [B, L, C]
        xs = x.chunk(self.win_cnt, -1)
        x_groups = []

        for block, x_chunk, mask in zip(self.Swin_block, xs, masks):
            x_chunk = block(x_chunk, mask)
            x_groups.append(x_chunk)

        x = torch.cat(x_groups, -1)  # [B,HW,C]  # 这里后面考虑改不改残差

        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):

        B, L, C = x.shape
        H, W = int(np.sqrt(L)), int(np.sqrt(L))
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 如果之前pad过
        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size, window_size_list=[7, 14],
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.window_size_list = window_size_list
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2  # 将窗口向右以及向下偏移window_size // 2个像素

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_Adaptive(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                window_size_list=window_size_list,
                shift_size=0 if (i % 2 == 0) else self.shift_size,  # block必须成对使用
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask_list(self, x, H, W):
        window_size_list = [7, 14]
        masks = []
        for window_size in window_size_list:
            attn_mask = self.create_mask1(window_size, x, H, W)
            masks.append(attn_mask)

        return masks

    def create_mask1(self, window_size, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / window_size)) * window_size
        Wp = int(np.ceil(W / window_size)) * window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]

        h_slices = (slice(0, -window_size),
                    slice(-window_size, -window_size),
                    slice(-window_size, None))
        w_slices = (slice(0, -window_size),
                    slice(-window_size, -window_size),
                    slice(-window_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, window_size * window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]

        # 上面两个矩阵相减得到窗口间注意力得分。相同窗口的注意力得分为0，否则为不相同窗口之间的注意力得分。
        # 将不为0的注意力填充为-100，softmax后会变为0，这样就创建了遮挡其他窗口的掩码。
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]

        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]

        # 上面两个矩阵相减得到窗口间注意力得分。相同窗口的注意力得分为0，否则为不相同窗口之间的注意力得分。
        # 将不为0的注意力填充为-100，softmax后会变为0，这样就创建了遮挡其他窗口的掩码。
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_masks = self.create_mask_list(x, H, W)
        # attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_masks)
            else:
                x = blk(x, attn_masks)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            resolution=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.resolution = resolution

        if skip_channels != 0:
            self.skipblock = skipblock(in_channels + skip_channels, resolution, resolution)
        else:
            self.skipblock = None

        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip_dict=None):
        x = self.up(x)
        if self.resolution != 0:
            if self.resolution == 32:
                skip_dict['x12'] = torch.cat([x, skip_dict['x11'], skip_dict['x10']], 1)
                x = skip_dict['x12']
            elif self.resolution == 64:
                skip_dict['x03'] = torch.cat([x, skip_dict['x00'], skip_dict['x01'], skip_dict['x02']], 1)
                x = skip_dict['x03']

            x = self.skipblock(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SkipBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()

        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class bottleneck(nn.Module):
    def __init__(self, in_channels, resolution=16):
        super().__init__()
        self.skip_up_layer = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block = bottleneckblock(in_channels, resolution)

    def forward(self, x16, x8):
        x8 = self.skip_upsample(x8)

        x = torch.cat([x16, x8], 1)
        x = self.block(x)

        return x

    # 跳跃链接上采样层 [B,L,C] -> [B, H, W, C] -> [B, 4 * L, C]
    def skip_upsample(self, x):
        x = self.skip_up_layer(x)
        return x


class Sigmoid_SegmentationHead(nn.Sequential):  # Multi_scale

    def __init__(self, in_channels, out_channels, upsampling=1, kernel_size=3):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class result_fuse(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_classes = config["n_classes"]

        decoder_channels = config.decoder_channels
        in_channels = [config.head_channels] + list(decoder_channels[:])
        out_channels = [num_classes, num_classes, num_classes, num_classes, num_classes]
        resolutions = config["multi_resolutions"]

        blocks = [
            Sigmoid_SegmentationHead(in_ch, out_ch, config["multi_resolutions"][-1] / resolution) for
            in_ch, out_ch, resolution in
            zip(in_channels, out_channels, resolutions)
        ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x_upsample):
        features = []
        for x, block in zip(x_upsample, self.blocks):
            feature = block(x)
            features.append(feature)

        return features


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 瓶颈
        self.bottleneck = bottleneck(config.hidden_size, config.resolutions[0])

        decoder_channels = config.decoder_channels
        in_channels = [config.head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        resolutions = config.resolutions

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels  # [192, 96, 0, 0]
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch, resolution) for in_ch, out_ch, sk_ch, resolution in
            zip(in_channels, out_channels, skip_channels, resolutions)
        ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, skip_dict=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)

        x = self.bottleneck(skip_dict['x20'], x)

        # 存储跳跃连接特征图
        x_upsample = [x]

        for i, decoder_block in enumerate(self.blocks):
            x = decoder_block(x, skip_dict=skip_dict)
            x_upsample.append(x)

        return x, x_upsample


class DSFA_SwinNet(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=2, window_size_list=[7, 14],
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.2, attn_drop_rate=0.4, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.window_size_list = window_size_list

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            # 这里stage包括了block和Patch Merging;前三个stage都有Patch Merging，第四个没有，因此downsample里设置self.num_layers - 1
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                window_size_list=window_size_list,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                # 前三个stage都有Patch Merging，第四个没有
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)

        # 密集连接字典
        self.skip_dict = {}
        # 对跳跃连接的下层张量进行上采样
        self.skip_up_layer = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv64_1 = SkipBlock(288, 64)
        self.conv32 = SkipBlock(576, 128)
        self.conv64_2 = SkipBlock(288, 64)

        # 上采样
        self.config = self.get_config()
        self.decoder = DecoderCup(self.config)

        self.segmentation_head = result_fuse(self.config)

        self.apply(self._init_weights)

    def get_config(self):
        resolution = 256
        resolution_2 = resolution // 2
        resolution_4 = resolution // 4
        resolution_8 = resolution // 8
        resolution_16 = resolution // 16

        config = ml_collections.ConfigDict()
        config.hidden_size = 1152
        config.decoder_channels = (resolution_2, resolution_4, resolution_8, resolution_16)
        config.n_classes = self.num_classes

        # bottle
        config.head_channels = resolution

        # dense skip
        config.resolutions = [resolution_8, resolution_4, 0, 0]

        # 跳跃连接
        config.n_skip = 3  # 3
        config.skip_channels = [320, 224, 0, 0]
        config.multi_resolutions = [resolution_16, resolution_8, resolution_4, resolution_2, resolution]

        return config

    def reshape(self, x):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        return x

    # 跳跃链接上采样层 [B,L,C] -> [B, H, W, C] -> [B, 4 * L, C]
    def skip_upsample(self, x):
        x = self.skip_up_layer(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C]
        x, H, W = self.patch_embed(x)

        x = self.pos_drop(x)

        # 存储跳跃连接特征图
        x_downsample = []

        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1:  # 检查是否是最后一层
                feature = self.reshape(x)
                x_downsample.append(feature)
            x, H, W = layer(x, H, W)

        # 将下采样存储的中间值编码进字典
        for idx, tensor in enumerate(x_downsample):
            self.skip_dict['x{}0'.format(idx)] = x_downsample[idx]

        # 计算中间值

        self.skip_dict['x01'] = torch.cat([self.skip_dict['x00'], self.skip_upsample(self.skip_dict['x10'])], 1)
        self.skip_dict['x01'] = self.conv64_1(self.skip_dict['x01'])

        self.skip_dict['x11'] = torch.cat([self.skip_dict['x10'], self.skip_upsample(self.skip_dict['x20'])], 1)
        self.skip_dict['x11'] = self.conv32(self.skip_dict['x11'])

        self.skip_dict['x02'] = torch.cat(
            [self.skip_dict['x00'], self.skip_dict['x01'], self.skip_upsample(self.skip_dict['x11'])], 1)
        self.skip_dict['x02'] = self.conv64_2(self.skip_dict['x02'])

        x = self.norm(x)  # [B, L, C]
        # print_x(x)

        x, x_upsample = self.decoder(x, self.skip_dict)  # 跳跃连接需补充

        features = self.segmentation_head(x_upsample)

        return features
