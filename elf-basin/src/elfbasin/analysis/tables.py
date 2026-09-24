def generate_ablation_table(results_dict, out_path):
    """
    results_dict: dict mapping variant_name (str) to dict {'bleu': float, 'entropy': float}
    Generates a LaTeX table for Phase 4 ablation ladder.
    """
    latex_str = "\\begin{table}[h]\n\\centering\n"
    latex_str += "\\begin{tabular}{lcc}\n"
    latex_str += "\\hline\n"
    latex_str += "\\textbf{Method (De-En)} & \\textbf{BLEU} $\\uparrow$ & \\textbf{Entropy} $\\uparrow$ \\\\\n"
    latex_str += "\\hline\n"
    
    # Define descriptive names for the variants
    names = {
        "v0": "0. Frozen Native Decoder (Baseline)",
        "v1": "1. Adapter (Clean-State CE)",
        "v2": "2. Adapter (Isotropic-Gaussian CE)",
        "v3": "3. Adapter (Scalar Hinge Margin)",
        "v4": "4. Adapter (Trajectory-Residual CE)",
        "v5": "5. \\quad + Sensitivity Penalty",
        "v6": "6. \\quad + Consistency KL"
    }
    
    for v in ["v0", "v1", "v2", "v3", "v4", "v5", "v6"]:
        if v in results_dict:
            res = results_dict[v]
            name = names.get(v, v)
            latex_str += f"{name} & {res['bleu']:.2f} & {res['entropy']:.3f} \\\\\n"
            
    latex_str += "\\hline\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\caption{Ablation of Sensitivity-Regularized Decoder Adapter.}\n"
    latex_str += "\\label{tab:ablation}\n"
    latex_str += "\\end{table}\n"
    
    with open(out_path, "w") as f:
        f.write(latex_str)
        
    return latex_str
