import os
import json
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .train import (
    get_device,
    get_baseline_toy_unet,
    get_rmcd_toy_unet,
    sample_images,
    measure_inference_peak_mem,
)

RESEARCH_DIR = os.path.join(".research", "iteration1")
IMAGES_DIR = os.path.join(RESEARCH_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


def experiment2_scaling_ablations(sizes: List[int] = None, steps: int = 8, device: Optional[torch.device] = None):
    device = device or get_device()
    sizes = sizes or [64, 96, 128, 192, 256]

    variants = {
        "RMCD(full)": dict(reversible=True, microchunk=True, dsg=True),
        "-Rev": dict(reversible=False, microchunk=True, dsg=True),
        "-DSG": dict(reversible=True, microchunk=True, dsg=False),
        "-Chunk": dict(reversible=True, microchunk=False, dsg=True),
    }

    table: Dict[str, Dict[int, Dict[str, Any]]] = {k: {} for k in variants}

    for vname, cfg in variants.items():
        print(f"[Experiment 2] Variant {vname}...")
        ctor = lambda: get_rmcd_toy_unet(base_ch=32, depth=2, lora_fuse=False, **cfg).to(device).eval()
        for s in sizes:
            model = ctor()
            def fwd():
                sample_images(model, n=1, size=s, steps=steps, device=device)
            try:
                peak, itps = measure_inference_peak_mem(fwd, warmup=1, iters=2, device=device)
                res = {"peak_mem_mb": float(peak), "it_per_s": float(itps), "oom": False}
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    res = {"peak_mem_mb": None, "it_per_s": None, "oom": True}
                else:
                    raise
            table[vname][s] = res
            print(f"  {vname}@{s}: {res}")

    with open(os.path.join(RESEARCH_DIR, 'exp2_scaling_ablation.json'), 'w') as f:
        json.dump(table, f, indent=2)

    # Plot memory scaling
    plt.figure(figsize=(6, 4))
    for vname in variants:
        ys = [table[vname][s]["peak_mem_mb"] if not table[vname][s]["oom"] else np.nan for s in sizes]
        plt.plot(sizes, ys, marker='o', label=vname)
    plt.xlabel('Resolution (H=W)')
    plt.ylabel('Peak mem (MB)')
    plt.title('Scaling of peak memory vs resolution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'scaling_peak_mem_rmcd_ablation.pdf'), bbox_inches='tight')
    plt.close()

    # Plot throughput scaling
    plt.figure(figsize=(6, 4))
    for vname in variants:
        ys = [table[vname][s]["it_per_s"] if not table[vname][s]["oom"] else 0.0 for s in sizes]
        plt.plot(sizes, ys, marker='s', label=vname)
    plt.xlabel('Resolution (H=W)')
    plt.ylabel('Iterations/sec (sampling)')
    plt.title('Scaling of iterations/sec vs resolution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'scaling_throughput_rmcd_ablation.pdf'), bbox_inches='tight')
    plt.close()

    print("[Experiment 2] Results saved to .research/iteration1/exp2_scaling_ablation.json and PDF figures in .research/iteration1/images.")


def experiment3_robustness_and_dsg(size_4k: int = 512, steps: int = 10, device: Optional[torch.device] = None):
    device = device or get_device()
    configs = {
        "Baseline": lambda: get_baseline_toy_unet(base_ch=32, depth=2).to(device).eval(),
        "RMCD(full)": lambda: get_rmcd_toy_unet(reversible=True, microchunk=True, dsg=True, base_ch=32, depth=2).to(device).eval(),
        "RMCD(-Chunk)": lambda: get_rmcd_toy_unet(reversible=True, microchunk=False, dsg=True, base_ch=32, depth=2).to(device).eval(),
    }

    report: Dict[str, Any] = {}
    for name, ctor in configs.items():
        print(f"[Experiment 3] Robustness @ {size_4k} for {name}")
        model = ctor()
        def fwd():
            sample_images(model, n=1, size=size_4k, steps=steps, device=device)
        try:
            peak, itps = measure_inference_peak_mem(fwd, warmup=1, iters=1, device=device)
            res = {"peak_mem_mb": float(peak), "it_per_s": float(itps), "oom": False}
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                res = {"oom": True}
            else:
                raise
        report[name] = res
        print("  ", name, res)

    with open(os.path.join(RESEARCH_DIR, 'exp3_robustness.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # Plot bar chart for robustness peak memory
    names = list(report.keys())
    mems = [report[k].get("peak_mem_mb", 0.0) if not report[k].get("oom", False) else 0.0 for k in names]
    colors = ['red' if report[k].get('oom', False) else 'green' for k in names]
    plt.figure(figsize=(5, 3))
    sns.barplot(x=names, y=mems, palette=colors)
    plt.ylabel('Peak mem (MB)')
    plt.title(f'Robustness peak memory @ {size_4k}^2')
    for i, name in enumerate(names):
        if report[name].get('oom', False):
            plt.text(i, 0.05, 'OOM', ha='center', va='bottom', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'robustness_peak_mem_4k.pdf'), bbox_inches='tight')
    plt.close()

    # DSG timeline profiling (mask fraction vs timestep)
    print("[Experiment 3] Profiling DSG timeline (mask fraction vs timestep) ...")
    model = get_rmcd_toy_unet(reversible=True, microchunk=True, dsg=True, base_ch=32, depth=2).to(device).eval()
    dsg_log: List[Dict[str, Any]] = []
    if hasattr(model, 'register_dsg_hook'):
        model.register_dsg_hook(lambda info: dsg_log.append(info))
    sample_images(model, n=1, size=128, steps=steps, device=device)
    t_to_vals: Dict[int, List[float]] = {}
    for e in dsg_log:
        t_to_vals.setdefault(int(e['t']), []).append(float(e['mask_frac']))
    ts = sorted(t_to_vals.keys())
    vals = [float(np.mean(t_to_vals[t])) for t in ts]
    if len(ts) > 0:
        plt.figure(figsize=(5, 3))
        plt.plot(ts, vals, marker='o')
        plt.xlabel('Timestep t')
        plt.ylabel('Mean mask fraction')
        plt.title('DSG mask timeline')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'dsg_mask_timeline_rmcd.pdf'), bbox_inches='tight')
        plt.close()
        with open(os.path.join(RESEARCH_DIR, 'dsg_timeline.json'), 'w') as f:
            json.dump({int(t): float(v) for t, v in zip(ts, vals)}, f, indent=2)
        print("[Experiment 3] DSG timeline saved.")
    else:
        print("[Experiment 3] DSG hook produced no data (check DSG implementation).")
