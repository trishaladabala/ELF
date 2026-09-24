import os
import sys
import json
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..', '..', '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

def load_results(directory):
    results = {}
    if not os.path.exists(directory):
        return results
    for f in os.listdir(directory):
        if f.endswith('_result.json'):
            path = os.path.join(directory, f)
            try:
                with open(path, 'r') as fp:
                    results[f] = json.load(fp)
            except Exception as e:
                print(f"Failed to load {f}: {e}")
    return results

def aggregate_metric(results_dict, condition_prefix, metric='gen_ppl'):
    vals = []
    for f, data in results_dict.items():
        if f.startswith(condition_prefix) and metric in data:
            vals.append(data[metric])
    if vals:
        return np.mean(vals), np.std(vals) / np.sqrt(len(vals))
    return None, None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='validation_phase')
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}.")
        return

    conditions = [
        ("1. Removing local time", [
            ("global_time_only", "Global Time Only (No Local Time)"),
        ]),
        ("2. Compressing local time", [
            ("frozen_K1", "Frozen K=1 (Constant τ)"),
            ("frozen_K2", "Frozen K=2 (from continuous)"),
            ("frozen_K4", "Frozen K=4 (from continuous)"),
        ]),
        ("3. Learning a compressed representation", [
            ("learned_quantized_K2", "Learned Quantized K=2"),
            ("learned_quantized_K4", "Learned Quantized K=4"),
            ("learned_lowrank_K2", "Learned Low-Rank K=2"),
            ("learned_lowrank_K4", "Learned Low-Rank K=4"),
        ]),
        ("4. Preserving full continuous local time", [
            ("continuous_local_time", "Continuous Local Time (Faithful Expansion)"),
        ])
    ]

    print("==============================================================")
    print(" EFM VALIDATION PHASE SUMMARY")
    print("==============================================================\n")

    for category, conds in conditions:
        print(f"### {category}")
        for prefix, label in conds:
            mean, stderr = aggregate_metric(results, prefix, 'gen_ppl')
            if mean is not None:
                print(f"- **{label}**: Gen-PPL = {mean:.2f} ± {stderr:.2f}")
            else:
                print(f"- **{label}**: (Pending)")
        print()

if __name__ == "__main__":
    main()

