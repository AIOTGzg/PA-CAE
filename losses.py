# MIT License
# 
# Copyright (c) 2018 Junyu Gao
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn


class DmdMreLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.0, eps=1e-6):
        super(DmdMreLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred_map, gt_map):
        if pred_map.shape != gt_map.shape:
            raise ValueError(
                f"pred_map and gt_map must have the same shape, got {pred_map.shape} and {gt_map.shape}"
            )

        if pred_map.dim() == 2:
            pred_map = pred_map.unsqueeze(0)
            gt_map = gt_map.unsqueeze(0)
        elif pred_map.dim() == 3:
            pred_map = pred_map.unsqueeze(1)
            gt_map = gt_map.unsqueeze(1)

        diff = pred_map - gt_map

        dmd_loss = 0.5 * diff.flatten(1).pow(2).sum(dim=1).mean()

        pred_cnt = pred_map.flatten(1).sum(dim=1)
        gt_cnt = gt_map.flatten(1).sum(dim=1)

        mre_loss = ((pred_cnt - gt_cnt).abs() / (gt_cnt.abs() + self.eps)).mean()

        total_loss = self.alpha * dmd_loss + self.beta * mre_loss
        return total_loss, dmd_loss, mre_loss
