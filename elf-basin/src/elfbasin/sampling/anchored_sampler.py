import torch
import torch.nn as nn
from utils.sampling_utils import _sde_step, _ode_step, restore_cond

def anchored_resume_generate(
    model, wrapper, z, start_step_idx, t_steps, cond_seq, cond_seq_mask, config, 
    cfg_scale=1.0, self_cond_cfg_scale=1.0, generator=None,
    intervention_type="none", # "none", "freeze", "reencode"
    anchor_positions=None,    # boolean mask (B, L)
    blend_lambda=1.0
):
    """
    Resumes generation from a given latent `z` at `start_step_idx`.
    Applies the specified intervention to the `anchor_positions` at the start,
    and then continues generation.
    """
    B, L, D = z.shape
    step_kwargs = dict(
        model=model, config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    
    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
    
    # ── Intervention ──
    if intervention_type != "none" and anchor_positions is not None:
        # We need to know x_hat to intervene. Let's do a dummy forward to get x_hat.
        # But wait, we can just get x_hat by running one ODE step with t_next=t_val (dt=0)
        t_val = t_steps[start_step_idx].item()
        _, x_hat = _ode_step(
            z=z, t=t_val, t_next=t_val, x_pred_prev=x_pred,
            **step_kwargs
        )
        
        if intervention_type == "reencode":
            # Decode to tokens
            logits = wrapper.decode(x_hat, sc_cfg=self_cond_cfg_scale)
            ids = logits.argmax(dim=-1)
            # Re-encode
            x_tilde = wrapper.encode(ids)
            
            # Blend
            x_hat_new = x_hat.clone()
            mask = anchor_positions.unsqueeze(-1).float()
            x_hat_new = (1 - blend_lambda * mask) * x_hat + (blend_lambda * mask) * x_tilde
            
            # Re-derive z consistently
            # z_t = t * x + (1-t) * eps -> if x changes, we adjust z by t * delta_x
            delta_x = x_hat_new - x_hat
            z = z + t_val * delta_x
            x_pred = x_hat_new
            
            frozen_x_hat = x_hat_new.clone()
            
        elif intervention_type == "freeze":
            frozen_x_hat = x_hat.clone()
            x_pred = frozen_x_hat
            
    # ── Generation Loop ──
    for i in range(start_step_idx, len(t_steps) - 1):
        t_val = t_steps[i].item()
        t_next = t_steps[i + 1].item()
        
        # Enforce freeze at every step
        if intervention_type == "freeze" and anchor_positions is not None:
            # For frozen positions, x_hat is fixed, so z_t must follow the ODE deterministically?
            # Actually, if x_hat is fixed, we can just force the network's x_pred to be frozen_x_hat
            pass
            
        # Standard step
        # Assuming ODE for simplicity (De-En uses ODE)
        z, x_pred_new = _ode_step(
            z=z, t=t_val, t_next=t_next, x_pred_prev=x_pred,
            **step_kwargs
        )
        
        if intervention_type in ["freeze", "reencode"] and anchor_positions is not None:
            # Force the prediction to remain the anchored one
            mask = anchor_positions.unsqueeze(-1).float()
            x_pred_new = (1 - mask) * x_pred_new + mask * frozen_x_hat
            
            # Force z to be consistent with the frozen x_hat?
            # Actually, standard diffusion forcing replaces z_t with the scheduled noisy version of x_ref.
            # z_next = t_next * x_ref + (1-t_next) * eps
            # If we don't know eps, we can't easily jump z. 
            # But wait, in the Flow ODE: dz = (x - z)/(1-t) dt
            # If we enforce x_pred_new, we should re-derive z?
            # Let's just override z by doing the Euler step analytically for the frozen parts:
            dz = (frozen_x_hat - z) / (1 - t_val + 1e-5) * (t_val - t_next) # note t goes from 1 to 0, so dt is negative
            # Actually _ode_step does z_next = z_t + (x_pred - z_t)/(t - 1) * (t_next - t)
            # So if we just set x_pred_new, the NEXT step will use it.
            # We must override z right now:
            z_override = z + (frozen_x_hat - z) / (t_val - 1 + 1e-5) * (t_next - t_val)
            z = (1 - mask) * z + mask * z_override
            
        x_pred = x_pred_new
        
    return z, x_pred
