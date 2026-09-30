"""Multi-head U-Net for charge-stability diagrams.

Dense heads (per pixel):
  occ_a, occ_b : electrons in dot a / dot b, classes 0,1,2,3,4+
  lines        : 5 sigmoid channels, one per line family (a, b, interdot, spectator, sensor)
Global heads (per scan):
  status       : FOUND / NOT_IN_WINDOW / UNINTERPRETABLE
  reason       : reason code (see schema.REASONS)
  ref          : is dot a / dot b anchored (empty region visible)? Occupancy is only
                 trustworthy for a dot whose ref is true; the loss is masked the same way.

Deliberately small (~0.5-2 M parameters): the data are simple images, the labels are precise,
and several independently trained copies (an ensemble) buy more reliability than one big net.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import schema


def _block(cin: int, cout: int) -> nn.Sequential:
    g = max(1, min(8, cout // 4))
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(g, cout), nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(g, cout), nn.SiLU(inplace=True),
    )


class ChargeCellNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 16, depth: int = 4):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chans:
            self.enc.append(_block(c, ch))
            c = ch
        self.mid = _block(c, c * 2)
        self.dec = nn.ModuleList()
        c = c * 2
        for ch in reversed(chans):
            self.dec.append(_block(c + ch, ch))
            c = ch
        n_occ = schema.N_OCC_CLASSES
        self.head_dense = nn.Conv2d(c, 2 * n_occ + len(schema.LINE_FAMILIES), 1)
        g_in = 2 * chans[-1] * 2 + 2 * c
        self.head_global = nn.Sequential(
            nn.Linear(g_in, 128), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, len(schema.STATUSES) + len(schema.REASONS) + 2),
        )
        self.config = dict(in_ch=in_ch, base=base, depth=depth)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        g_mid = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
        for blk, s in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, s], 1))
        dense = self.head_dense(x)
        n = schema.N_OCC_CLASSES
        g = self.head_global(torch.cat([g_mid, x.mean((2, 3)), x.amax((2, 3))], 1))
        ns, nr = len(schema.STATUSES), len(schema.REASONS)
        return {
            "occ_a": dense[:, :n], "occ_b": dense[:, n:2 * n], "lines": dense[:, 2 * n:],
            "status": g[:, :ns], "reason": g[:, ns:ns + nr], "ref": g[:, ns + nr:],
        }


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
