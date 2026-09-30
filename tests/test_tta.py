"""Test-time augmentation: the views are exact symmetries of the features, and the outputs of the
axis-swapped views are mapped back to the right dots."""
import numpy as np
import torch
from torch import nn

from chargecell import kinds
from chargecell.model.train import predict_batch
from chargecell.preprocess import features


def test_feature_symmetries():
    rng = np.random.default_rng(0)
    sig = np.cumsum(rng.normal(size=(40, 40)), 1) + 0.3 * rng.normal(size=(40, 40))
    f = features(sig)
    assert np.allclose(features(-sig), -f, atol=1e-5)
    assert np.allclose(features(sig.T.copy()), np.stack([f[0].T, f[2].T, f[1].T]), atol=1e-5)


class _Equivariant(nn.Module):
    """A stand-in network that treats the dots correctly by construction: dot a is read from the
    x derivative and x position, dot b from y. Polarity-invariant (even in the input). With a
    correct mapping of the swapped views, TTA must return exactly its plain output."""

    def __init__(self):
        super().__init__()
        self.spec = kinds.get("PvP")

    def forward(self, x):
        z2, gx2, gy2 = x[:, 0] ** 2, x[:, 1] ** 2, x[:, 2] ** 2
        S = x.shape[-1]
        col = torch.arange(S, dtype=x.dtype).expand(S, S) / S       # index along x
        row = col.T                                                  # index along y
        k = torch.arange(5, dtype=x.dtype)[None, :, None, None]
        fam = {"a": gx2, "b": gy2}
        lines = torch.stack([fam.get(f, z2 + gx2 + gy2) for f in self.spec.line_families], 1)
        ma, mb, mz = gx2.mean((1, 2)), gy2.mean((1, 2)), z2.mean((1, 2))
        n_reasons = len(self.spec.reasons)
        glob = torch.stack([ma + mb, mz, ma * mb], 1)
        return {"occ_a": k * col + gx2[:, None] * (k - 2),
                "occ_b": k * row + gy2[:, None] * (k - 2),
                "lines": lines,
                "status": glob,
                "reason": (ma + mb + mz)[:, None] * torch.linspace(-1, 1, n_reasons),
                "ref": torch.stack([ma - mz, mb - mz], 1)}


def test_tta_maps_swapped_views_back_to_the_right_dot():
    rng = np.random.default_rng(1)
    sig = np.cumsum(rng.normal(size=(24, 24)), 1) + np.cumsum(rng.normal(size=(24, 24)), 0) * 0.3
    x = torch.from_numpy(features(sig))[None]
    m = _Equivariant()
    plain, tta = predict_batch([m], x), predict_batch([m], x, tta=True)
    assert tta["status_members"].shape[0] == 4           # identity, polarity, swap, both
    for key in ("occ_a", "occ_b", "lines", "status", "reason", "ref"):
        assert torch.allclose(plain[key], tta[key], atol=1e-5), key
    # and dot a really differs from dot b here, so a wrong mapping would have shown
    assert not torch.allclose(plain["occ_a"], plain["occ_b"], atol=1e-3)
    assert not torch.allclose(plain["ref"][:, 0], plain["ref"][:, 1], atol=1e-3)
