import os
import json

out_dir = "task2a_results"
Ws = [4, 16]
sigmas = [0.0, 0.05, 0.1, 0.2]
conditions = ["global_time_only", "continuous", "lowrank_K4"]

print("="*60)
print("TEST 2a: TF EXACT MATCH RESULTS")
print("="*60)

for W in Ws:
    print(f"\nTask Complexity: W={W}")
    print(f"{'Sigma':<10} {'global_time':<15} {'continuous':<15} {'lowrank_K4':<15}")
    print("-" * 55)
    for sigma in sigmas:
        row = [f"{sigma:<10}"]
        for name in conditions:
            acc = 0.0
            if os.path.exists(os.path.join(out_dir, "summary.json")):
                with open(os.path.join(out_dir, "summary.json")) as f:
                    data = json.load(f)
                    if str(W) in data and str(sigma) in data[str(W)] and name in data[str(W)][str(sigma)]:
                        acc = data[str(W)][str(sigma)][name]["acc"]
            row.append(f"{acc:<15.4f}")
        print(" ".join(row))
