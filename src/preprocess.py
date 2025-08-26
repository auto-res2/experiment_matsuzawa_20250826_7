import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
from typing import Optional, List, Tuple

class SyntheticImageDataset(Dataset):
    """
    Generates multiple data patterns to stress-test robustness.
    Patterns: 'noise', 'checker', 'grad', 'stripes'
    Returns image in [-1, 1] and a dummy text conditioning embedding [77, 768].
    """
    def __init__(self, n: int = 1024, size: int = 128, patterns: Optional[List[str]] = None):
        self.n = n
        self.size = size
        self.patterns = patterns or ["noise", "checker", "grad", "stripes"]

    def __len__(self):
        return self.n

    def _make_noise(self):
        img = torch.rand(3, self.size, self.size)
        return img

    def _make_checker(self):
        s = self.size
        x = torch.arange(s).unsqueeze(0).repeat(s, 1)
        y = torch.arange(s).unsqueeze(1).repeat(1, s)
        board = ((x // 8 + y // 8) % 2).float()
        img = board.unsqueeze(0).repeat(3, 1, 1)
        return img

    def _make_grad(self):
        s = self.size
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing='ij')
        rr = torch.sqrt(xx**2 + yy**2)
        img = (1 - rr).clamp(0, 1)
        img = img.unsqueeze(0).repeat(3, 1, 1)
        return img

    def _make_stripes(self):
        s = self.size
        yy = torch.arange(s).float().unsqueeze(1).repeat(1, s)
        stripes = ((yy // 6) % 2).float()
        img = stripes.unsqueeze(0).repeat(3, 1, 1)
        return img

    def _make_img(self, pattern):
        if pattern == "noise":
            return self._make_noise()
        elif pattern == "checker":
            return self._make_checker()
        elif pattern == "grad":
            return self._make_grad()
        elif pattern == "stripes":
            return self._make_stripes()
        else:
            return self._make_noise()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pattern = random.choice(self.patterns)
        img = self._make_img(pattern)
        img = img * 2 - 1
        cond = torch.randn(77, 768)
        return img, cond


class SyntheticVideoDataset(Dataset):
    def __init__(self, n: int = 256, T: int = 8, H: int = 64, W: int = 64):
        self.n, self.T, self.H, self.W = n, T, H, W

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vid = (torch.rand(3, self.T, self.H, self.W) * 2 - 1)
        cond = torch.randn(77, 768)
        return vid, cond
