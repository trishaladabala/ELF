import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_loss(adapter, x_hat, r, r_prime, v, y, lambda_sens, lambda_cons, sc_cfg=1.0):
    """
    Computes the Phase 4 training objective:
    L = CE( decode(x̂ + r), y )                                  # trajectory-realistic CE
      + λ_sens · E_v [ ‖ ∇_x (g_top1 − g_top2) · v ‖ ]           # local sensitivity penalty
      + λ_cons · KL( decode(x̂) ‖ decode(x̂ + r') )                # neighborhood consistency
    
    y: (B, L) target tokens
    x_hat: (B, L, D) base latent
    r, r_prime: (B, L, D) residuals sampled from empirical distribution
    v: (B, L, D) perturbation direction sampled from empirical trajectory distribution
    """
    # 1. Trajectory-realistic CE
    x_noisy = x_hat + r
    logits_noisy = adapter(x_noisy, sc_cfg=sc_cfg) # (B, L, V)
    
    # Cross entropy wants (B*L, V) and (B*L,)
    ce_loss = F.cross_entropy(
        logits_noisy.reshape(-1, logits_noisy.size(-1)),
        y.reshape(-1),
        ignore_index=0 # assuming 0 is padding (eos for T5 usually)
    )
    
    # 2. Local sensitivity penalty (m/L regularization)
    sens_loss = torch.tensor(0.0, device=x_hat.device)
    if lambda_sens > 0:
        # We need to compute gradient of margin wrt x_noisy.
        x_noisy_req = x_noisy.detach().clone()
        x_noisy_req.requires_grad_(True)
        
        # Flash attention doesn't support double backwards, so we must force math SDP
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            logits_req = adapter(x_noisy_req, sc_cfg=sc_cfg)
            # Get top 2 logits for margin
            top2 = logits_req.topk(2, dim=-1).values
            margin = top2[:, :, 0] - top2[:, :, 1]
            
            # We want the directional derivative along v: ∇_x(margin) · v
            # Which is equivalent to taking the gradient of (margin * v).sum() ? No.
            # It's (∇_x margin) · v. We can compute this using jvp or standard backward.
            grad_x = torch.autograd.grad(
                outputs=margin,
                inputs=x_noisy_req,
                grad_outputs=torch.ones_like(margin),
                create_graph=True,
                retain_graph=True
            )[0]
        
        # Dot product with v, then norm (abs value)
        dir_deriv = (grad_x * v).sum(dim=-1).abs()
        sens_loss = dir_deriv.mean()
        
    # 3. Neighborhood consistency KL
    cons_loss = torch.tensor(0.0, device=x_hat.device)
    if lambda_cons > 0:
        with torch.no_grad():
            logits_clean = adapter(x_hat, sc_cfg=sc_cfg)
            
        logits_noisy_prime = adapter(x_hat + r_prime, sc_cfg=sc_cfg)
        
        # KL Divergence: KL( clean || noisy_prime )
        log_probs_prime = F.log_softmax(logits_noisy_prime, dim=-1)
        probs_clean = F.softmax(logits_clean, dim=-1)
        
        # Only compute over valid tokens (y != 0)
        valid = (y != 0)
        kl = F.kl_div(log_probs_prime[valid], probs_clean[valid], reduction='batchmean')
        cons_loss = kl
        
    total_loss = ce_loss + lambda_sens * sens_loss + lambda_cons * cons_loss
    return total_loss, ce_loss.detach(), sens_loss.detach(), cons_loss.detach()
