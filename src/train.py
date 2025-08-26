import os
import time
import json
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocess import SyntheticImageDataset

# Global paths
RESEARCH_DIR = os.path.join(".research", "iteration1")
IMAGES_DIR = os.path.join(RESEARCH_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

SEED = 1337


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class BenchmarkResult:
    peak_mem_mb: float
    imgs_per_sec: float
    loss: float
    notes: Dict[str, Any]


# ------------------------------
# Diffusion trainer and samplers
# ------------------------------
class OutputWrap:
    def __init__(self, sample: torch.Tensor):
        self.sample = sample


class FiLM(nn.Module):
    def __init__(self, c: int, cond_dim: int = 768, t_dim: int = 64):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim + t_dim, c)
        self.to_beta = nn.Linear(cond_dim + t_dim, c)
        self.t_emb = nn.Embedding(1000, t_dim)

    def forward(self, h: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        cond_vec = cond.mean(dim=1)
        t_vec = self.t_emb(t.clamp(min=0, max=999))
        fused = torch.cat([cond_vec, t_vec], dim=-1)
        gamma = self.to_gamma(fused).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(fused).unsqueeze(-1).unsqueeze(-1)
        return h * (1 + gamma.tanh()) + beta


class ResidualBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, c)
        self.norm2 = nn.GroupNorm(8, c)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + x)


class ReversibleCoupling(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        assert c % 2 == 0, "Channels must be even for reversible split"
        self.f = ResidualBlock(c // 2)
        self.g = ResidualBlock(c // 2)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return torch.cat([y1, y2], dim=1)


class MicroChunker:
    def __init__(self, chunk_h: int = 32, chunk_w: int = 32):
        self.chunk_h = chunk_h
        self.chunk_w = chunk_w

    def apply(self, fn, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ch, cw = self.chunk_h, self.chunk_w
        if H <= ch and W <= cw:
            return fn(x)
        out = torch.empty_like(x)
        for y in range(0, H, ch):
            for z in range(0, W, cw):
                patch = x[:, :, y:min(y+ch, H), z:min(z+cw, W)]
                out[:, :, y:min(y+ch, H), z:min(z+cw, W)] = fn(patch)
        return out


class DSGate(nn.Module):
    def __init__(self, tile: int = 16):
        super().__init__()
        self.tile = tile
        self.score = nn.Conv2d(3, 1, 1)
        self.t_emb = nn.Embedding(1000, 8)
        self.proj = nn.Linear(8, 1)
        self._hook = None

    def register_hook(self, hook_fn):
        self._hook = hook_fn

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = self.score(x).mean(dim=0, keepdim=True)
        tile = self.tile
        HH = (H + tile - 1) // tile
        WW = (W + tile - 1) // tile
        pooled = F.adaptive_avg_pool2d(s, (HH, WW))
        pooled = pooled.expand(B, -1, -1, -1)
        t_feat = self.proj(self.t_emb(t.clamp(0, 999))).view(B, 1, 1, 1)
        thr = torch.sigmoid(t_feat)
        mask_tiles = (torch.sigmoid(pooled) > thr).float()
        mask = F.interpolate(mask_tiles, size=(H, W), mode='nearest')
        if self._hook is not None:
            mask_frac = float(mask_tiles.mean().detach().cpu().item())
            self._hook({"t": int(t.float().mean().item()), "mask_frac": mask_frac})
        return mask


class LoRAConvWrapper(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int = 4):
        super().__init__()
        # Freeze base conv
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_c = base.in_channels
        out_c = base.out_channels
        k = base.kernel_size[0]
        pad = base.padding
        stride = base.stride
        dilation = base.dilation
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        r = max(1, min(rank, max(1, in_c // 8)))
        self.lora_A = nn.Conv2d(in_c, r, k, padding=pad, stride=stride, dilation=dilation, bias=False)
        self.lora_B = nn.Conv2d(r, out_c, 1, bias=False)

    def forward(self, x):
        y = self.base(x)
        a = self.lora_A(x)
        b = self.lora_B(a)
        return y + b


class SimpleUNetLike(nn.Module):
    def __init__(self, base_ch: int = 64, depth: int = 3, reversible: bool = False,
                 microchunk: bool = False, dsg: bool = False, lora_fuse: bool = False):
        super().__init__()
        self.reversible = reversible
        self.microchunk = microchunk
        self.dsg_flag = dsg
        self.lora_fuse = lora_fuse

        self.in_conv = nn.Conv2d(3, base_ch, 3, padding=1)
        self.film = FiLM(base_ch)

        ch = base_ch
        self.downs = nn.ModuleList()
        for _ in range(depth):
            block = ReversibleCoupling(ch) if reversible and (ch % 2 == 0) else ResidualBlock(ch)
            self.downs.append(block)
            self.downs.append(nn.Conv2d(ch, ch*2, 3, stride=2, padding=1))
            ch *= 2
        self.mid = ResidualBlock(ch)
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(nn.ConvTranspose2d(ch, ch//2, 2, stride=2))
            ch //= 2
            block = ReversibleCoupling(ch) if reversible and (ch % 2 == 0) else ResidualBlock(ch)
            self.ups.append(block)
        self.out_conv = nn.Conv2d(ch, 3, 3, padding=1)

        self.chunker = MicroChunker(chunk_h=32, chunk_w=32)
        self.dsg = DSGate(tile=16) if dsg else None
        if self.lora_fuse:
            self._apply_lora_replace()

    def _apply_lora_replace(self):
        def replace(module: nn.Module):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Conv2d):
                    setattr(module, name, LoRAConvWrapper(child))
                else:
                    replace(child)
        replace(self)

    def register_dsg_hook(self, hook_fn):
        if self.dsg is not None:
            self.dsg.register_hook(hook_fn)

    def _apply_microchunk(self, fn, x):
        if self.microchunk:
            return self.chunker.apply(fn, x)
        else:
            return fn(x)

    def forward(self, x: torch.Tensor, t: torch.Tensor, encoder_hidden_states: torch.Tensor) -> OutputWrap:
        if self.dsg is not None:
            mask = self.dsg(x, t)
        else:
            mask = None

        def enc_block(inp):
            h = self.in_conv(inp)
            h = self.film(h, encoder_hidden_states, t)
            skips = []
            for i in range(0, len(self.downs), 2):
                block, down = self.downs[i], self.downs[i+1]
                h = block(h)
                skips.append(h)
                h = down(h)
            h = self.mid(h)
            for i in range(0, len(self.ups), 2):
                up, block = self.ups[i], self.ups[i+1]
                h = up(h)
                if len(skips) > 0:
                    h = h + skips.pop()
                h = block(h)
            h = self.out_conv(h)
            return h

        if mask is None:
            out = self._apply_microchunk(enc_block, x)
        else:
            pred_full = self._apply_microchunk(enc_block, x)
            out = mask * pred_full + (1 - mask) * x
        return OutputWrap(sample=out)


def get_baseline_toy_unet(base_ch: int = 32, depth: int = 2) -> nn.Module:
    return SimpleUNetLike(base_ch=base_ch, depth=depth, reversible=False, microchunk=False, dsg=False, lora_fuse=False)


def get_rmcd_toy_unet(reversible: bool = True, microchunk: bool = True, dsg: bool = True, lora_fuse: bool = False,
                       base_ch: int = 32, depth: int = 2) -> nn.Module:
    return SimpleUNetLike(base_ch=base_ch, depth=depth, reversible=reversible, microchunk=microchunk,
                          dsg=dsg, lora_fuse=lora_fuse)


class SimpleDiffusionTrainer:
    def __init__(self, unet: nn.Module, timesteps: int = 1000):
        self.unet = unet
        self.timesteps = timesteps

    def sample_timesteps(self, b: int, device: torch.device):
        return torch.randint(0, self.timesteps, (b,), device=device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        view_shape = [x0.size(0)] + [1] * (x0.ndim - 1)
        alpha = (1 - (t.float() / max(1, (self.timesteps - 1)))).view(*view_shape)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(alpha) * x0 + torch.sqrt(1 - alpha) * eps
        return xt, eps

    def loss_step_images(self, x: torch.Tensor, cond: torch.Tensor, device: torch.device):
        b = x.size(0)
        t = self.sample_timesteps(b, device)
        xt, eps = self.add_noise(x, t)
        out = self.unet(xt, t, encoder_hidden_states=cond)
        pred = out.sample
        return F.mse_loss(pred, eps)


@torch.inference_mode()
def sample_images(unet: nn.Module, n: int = 8, size: int = 128, steps: int = 10, device: Optional[torch.device] = None):
    device = device or get_device()
    unet.eval().to(device)
    x = torch.randn(n, 3, size, size, device=device)
    cond = torch.randn(n, 77, 768, device=device)
    for s in reversed(range(steps)):
        t = torch.full((n,), s, device=device)
        out = unet(x, t, encoder_hidden_states=cond)
        eps = out.sample
        x = x - eps / max(1, steps)
    x = (x.clamp(-1, 1) + 1) / 2
    return x


@torch.inference_mode()
def measure_inference_peak_mem(forward_fn, warmup: int = 1, iters: int = 3, device: Optional[torch.device] = None):
    device = device or get_device()
    if device.type == "cpu":
        start = time.time()
        for _ in range(warmup):
            forward_fn()
        start = time.time()
        for _ in range(iters):
            forward_fn()
        elapsed = time.time() - start
        return 0.0, (iters / max(1e-9, elapsed))
    else:
        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            forward_fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for _ in range(iters):
            forward_fn()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        return float(peak), (iters / max(1e-9, elapsed))


def train_benchmark(model: nn.Module,
                    dataset,
                    steps: int = 50,
                    batch_size: int = 2,
                    lr: float = 1e-3,
                    amp: bool = False,
                    grad_accum: int = 1,
                    num_workers: int = 0,
                    device: Optional[torch.device] = None) -> BenchmarkResult:
    set_seed()
    device = device or get_device()
    model = model.to(device)
    model.train()

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    trainer = SimpleDiffusionTrainer(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == 'cuda'))

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    n_seen = 0
    start = time.time()
    it = iter(dl)
    loss_val = float('inf')

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            try:
                x, cond = next(it)
            except StopIteration:
                it = iter(dl)
                x, cond = next(it)
            x = x.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(amp and device.type == 'cuda')):
                loss = trainer.loss_step_images(x, cond, device)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            n_seen += x.size(0)
            loss_val = float(loss.item() * grad_accum)
        scaler.step(opt)
        scaler.update()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    imgs_per_sec = n_seen / max(1e-9, elapsed)
    peak = (torch.cuda.max_memory_allocated() / (1024**2)) if device.type == 'cuda' else 0.0

    return BenchmarkResult(peak_mem_mb=float(peak), imgs_per_sec=float(imgs_per_sec), loss=loss_val, notes={})


# ------------------------------
# Experiment 1: Train and inference benchmark (Images)
# ------------------------------

def experiment1_train_and_infer_bench(steps: int = 30, img_size: int = 128, batch_size: int = 2,
                                       amp: bool = False, device: Optional[torch.device] = None):
    device = device or get_device()
    print("[Experiment 1] Device:", device)
    dataset = SyntheticImageDataset(n=max(256, steps * batch_size), size=img_size)

    models = {
        "Baseline": get_baseline_toy_unet(base_ch=32, depth=2),
        "RMCD(full)": get_rmcd_toy_unet(reversible=True, microchunk=True, dsg=True, lora_fuse=False, base_ch=32, depth=2),
        "RMCD(-Rev)": get_rmcd_toy_unet(reversible=False, microchunk=True, dsg=True, lora_fuse=False, base_ch=32, depth=2),
    }

    results: Dict[str, Dict[str, float]] = {}
    loss_curves: Dict[str, List[float]] = {}

    for name, model in models.items():
        print(f"[Experiment 1] Training {name}...")
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        pin_memory=(device.type == 'cuda'))
        trainer = SimpleDiffusionTrainer(model)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == 'cuda'))
        model.to(device).train()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        it = iter(dl)
        losses: List[float] = []
        n_seen = 0
        start = time.time()
        for s in range(steps):
            try:
                x, cond = next(it)
            except StopIteration:
                it = iter(dl)
                x, cond = next(it)
            x = x.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and device.type == 'cuda')):
                loss = trainer.loss_step_images(x, cond, device)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            n_seen += x.size(0)
            losses.append(float(loss.item()))
            if (s+1) % max(1, (steps//5)) == 0:
                print(f"  Step {s+1}/{steps}: loss={loss.item():.4f}")
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        ips = n_seen / max(1e-9, elapsed)
        peak = (torch.cuda.max_memory_allocated() / (1024**2)) if device.type == 'cuda' else 0.0
        results[name] = {"peak_mem_mb": float(peak), "imgs_per_sec": float(ips), "final_loss": float(losses[-1])}
        loss_curves[name] = losses
        print(f"[Experiment 1] {name}: peak_mem={peak:.1f}MB, imgs/sec={ips:.2f}, final_loss={losses[-1]:.4f}")

    # Plot training loss curves (PDF)
    plt.figure(figsize=(6, 4))
    for name, losses in loss_curves.items():
        plt.plot(losses, label=name)
    plt.xlabel('Step')
    plt.ylabel('Training loss (MSE)')
    plt.title('Training loss (synthetic)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'training_loss_baselines.pdf'), bbox_inches='tight')
    plt.close()

    # Plot throughput
    names = list(results.keys())
    throughputs = [results[k]["imgs_per_sec"] for k in names]
    plt.figure(figsize=(5, 3))
    sns.barplot(x=names, y=throughputs, palette='viridis')
    plt.ylabel('Images/sec')
    plt.title('Training throughput')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'throughput_baselines.pdf'), bbox_inches='tight')
    plt.close()

    # Inference memory/speed
    infer_stats: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        model.eval().to(device)
        def fwd():
            sample_images(model, n=2, size=img_size, steps=5, device=device)
        peak, itps = measure_inference_peak_mem(fwd, warmup=1, iters=2, device=device)
        infer_stats[name] = {"peak_mem_mb": float(peak), "it_per_s": float(itps)}
        print(f"[Experiment 1] Inference {name}: peak_mem={peak:.1f}MB, it/s={itps:.2f}")

    # Plot inference peak memory
    names = list(infer_stats.keys())
    mems = [infer_stats[k]["peak_mem_mb"] for k in names]
    plt.figure(figsize=(5, 3))
    sns.barplot(x=names, y=mems, palette='magma')
    plt.ylabel('Peak mem (MB)')
    plt.title('Inference peak memory')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'inference_peak_mem_baselines.pdf'), bbox_inches='tight')
    plt.close()

    # Save JSON summary under .research/iteration1
    summary = {"train": results, "infer": infer_stats}
    with open(os.path.join(RESEARCH_DIR, 'exp1_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("[Experiment 1] Summary saved to .research/iteration1/exp1_summary.json; figures saved as PDFs in .research/iteration1/images.")
