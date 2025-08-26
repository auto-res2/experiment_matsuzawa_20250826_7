import argparse
import os
import yaml

from .train import experiment1_train_and_infer_bench, get_device, set_seed
from .evaluate import experiment2_scaling_ablations, experiment3_robustness_and_dsg


def run_from_config(cfg_path: str):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    mode = cfg.get('mode', 'quick')
    device = get_device()
    set_seed(1337)

    print("Running mode:", mode)
    if mode == 'quick':
        e1 = cfg.get('exp1', {})
        experiment1_train_and_infer_bench(
            steps=int(e1.get('steps', 8)),
            img_size=int(e1.get('img_size', 64)),
            batch_size=int(e1.get('batch_size', 2)),
            amp=bool(e1.get('amp', False)),
            device=device,
        )
        e2 = cfg.get('exp2', {})
        experiment2_scaling_ablations(
            sizes=list(e2.get('sizes', [48, 64, 80])),
            steps=int(e2.get('steps', 5)),
            device=device,
        )
        e3 = cfg.get('exp3', {})
        experiment3_robustness_and_dsg(
            size_4k=int(e3.get('size_4k', 256)),
            steps=int(e3.get('steps', 6)),
            device=device,
        )
    elif mode == 'full':
        e1 = cfg.get('exp1', {})
        experiment1_train_and_infer_bench(
            steps=int(e1.get('steps', 100)),
            img_size=int(e1.get('img_size', 128)),
            batch_size=int(e1.get('batch_size', 4)),
            amp=bool(e1.get('amp', False)),
            device=device,
        )
        e2 = cfg.get('exp2', {})
        experiment2_scaling_ablations(
            sizes=list(e2.get('sizes', [64, 96, 128, 192, 256])),
            steps=int(e2.get('steps', 10)),
            device=device,
        )
        e3 = cfg.get('exp3', {})
        experiment3_robustness_and_dsg(
            size_4k=int(e3.get('size_4k', 512)),
            steps=int(e3.get('steps', 10)),
            device=device,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMCD Toy Experiments Runner")
    parser.add_argument("--config", type=str, default=os.path.join("config", "config.yaml"), help="Path to config.yaml")
    args = parser.parse_args()

    run_from_config(args.config)
