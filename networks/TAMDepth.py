# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ops.modules.ms_deform_attn import MSDeformAttn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_
from functools import partial


from networks.smt import SMT
# from base.vit import TIMMVisionTransformer
from networks.adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs, Conv33

_logger = logging.getLogger(__name__)


class TAMDepth(SMT):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4, deform_num_heads=4,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_extractor=True, *args, **kwargs):

        super().__init__(num_heads=num_heads, *args, **kwargs)

        # self.cls_token = None
        # self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.add_vit_feature = add_vit_feature
        embed_dim = int(self.num_ch_enc[-2])

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=int(self.num_ch_enc[-2]))
        self.interactions = []
        for i in range(self.num_stages):
            self.interactions = nn.Sequential(*[
                InteractionBlock(dim=int(self.num_ch_enc[i]), num_heads=deform_num_heads, n_points=n_points,
                                 init_values=init_values, drop_path=self.drop_path_rate,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cffn=with_cffn,
                                 cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                                 extra_extractor=((True if i == 4 else False) and use_extra_extractor))
                for i in range(self.num_stages)])

        self.up_1 = nn.ConvTranspose2d(self.num_ch_enc[-1] * 2, self.num_ch_enc[0], 2, 2)
        self.up_2 = nn.ConvTranspose2d(self.num_ch_enc[-1] * 2, self.num_ch_enc[1], 2, 2)
        self.up_3 = nn.ConvTranspose2d(self.num_ch_enc[-1] * 2, self.num_ch_enc[2], 2, 2)
        self.up_4 = nn.ConvTranspose2d(self.num_ch_enc[-1] * 2, self.num_ch_enc[3], 1, 1)
        self.norm_1 = nn.BatchNorm2d(self.num_ch_enc[0])
        self.norm_2 = nn.BatchNorm2d(self.num_ch_enc[1])
        self.norm_3 = nn.BatchNorm2d(self.num_ch_enc[2])
        self.norm_4 = nn.BatchNorm2d(self.num_ch_enc[3])
        self.conv_3 = Conv33(self.num_ch_enc[-2])

        self.up_1.apply(self._init_weights)
        self.up_2.apply(self._init_weights)
        self.up_3.apply(self._init_weights)
        self.up_4.apply(self._init_weights)
        
        self.spm.apply(self._init_weights)
        self.conv_3.apply(self._init_weights)
        for i in range(self.num_stages):
            self.interactions[i].apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs_11, deform_inputs_21 = deform_inputs(x, 4)
        deform_inputs_12, deform_inputs_22 = deform_inputs(x, 8)
        deform_inputs1 = [deform_inputs_11, deform_inputs_12]
        deform_inputs2 = [deform_inputs_21, deform_inputs_22]
        x_vit = []
        B = x.shape[0]
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Interaction
        for i, layer in enumerate(self.interactions):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            if i == 2 or i == 3:
                x, c = layer(x, c, block,
                             deform_inputs1[i - 2], deform_inputs2[i - 2], H, W, i)
            else:
                for blk in block:
                    x = blk(x, H, W)
            if i == 2: c = self.conv_3(c, H, W, i)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x_vit.append(x)

        c1, c2, c3, c4 = x_vit[0], x_vit[1], x_vit[2], x_vit[3]

        # Final Norm
        f1 = self.norm_1(c1)
        f2 = self.norm_2(c2)
        f3 = self.norm_3(c3)
        f4 = self.norm_4(c4)
        return [f1, f2, f3, f4]


