import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False):

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps

        sigma, dsigma = noise(t)

        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss

    return loss_fn


def get_kl_loss_fn(noise, graph, train, sampling_eps=1e-3):
    """
    ELBO-based loss for the absorbing graph: minimizes the variational lower bound on
    -log p_theta(x_0), which is the tractable form of forward KL divergence KL(p_data || p_theta).

    For the absorbing (masking) graph, the forward process posterior q(x_{t-dt} | x_t, x_0) is
    analytically available:
      - If x_t is unmasked: q is a delta on x_t (no information to predict; KL = 0).
      - If x_t = [MASK]: q is a 2-point distribution over {x_0, [MASK]}:
            q_unmask = (e^{-sigma(t-dt)} - e^{-sigma(t)}) / (1 - e^{-sigma(t)})
            q_mask   = (1 - e^{-sigma(t-dt)}) / (1 - e^{-sigma(t)})
        In the continuous-time limit dt->0, this becomes:
            q_unmask -> dsigma * e^{-sigma} / (1 - e^{-sigma})  =  dsigma / (e^sigma - 1)
            q_mask   -> 1 - q_unmask
        The model predicts a categorical p_theta over all |V|+1 tokens via softmax on log-scores.
        KL(q || p_theta) = q_unmask * log(q_unmask / p_theta(x_0))
                         + q_mask   * log(q_mask   / p_theta([MASK]))

    The total loss integrates over t, weighted by dsigma (the noise schedule rate), mirroring
    the continuous-time ELBO derivation for score-based diffusion models.
    """
    if not graph.absorb:
        raise ValueError("KL loss is only implemented for the absorbing graph.")

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        if t is None:
            t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps

        sigma, dsigma = noise(t)          # [B], [B]
        sigma_b = sigma[:, None]          # [B, 1] for broadcasting over sequence

        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma_b)

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)  # [B, L, V+1] log-ratios

        # Convert log-scores to log-probabilities via log-softmax.
        # The model outputs log p(y) / p(x_t) for each y; softmax over y gives
        # the categorical p_theta(x_{t-dt} | x_t).
        log_probs = F.log_softmax(log_score, dim=-1)      # [B, L, V+1]

        # Only masked positions contribute to the loss.
        mask_idx = graph.dim - 1                           # index of [MASK] token
        is_masked = (perturbed_batch == mask_idx)          # [B, L] bool

        # Continuous-time posterior weights at masked positions.
        # q_unmask = dsigma / (e^sigma - 1), the instantaneous unmasking rate.
        # We use expm1 for numerical stability when sigma is small.
        esigm1 = torch.where(
            sigma_b < 0.5,
            torch.expm1(sigma_b),
            sigma_b.exp() - 1,
        )                                                  # [B, L]  =  e^sigma - 1

        # dsigma broadcast to [B, L]
        dsigma_b = dsigma[:, None].expand_as(perturbed_batch)

        # q_unmask: weight on the "correct token" branch of the posterior
        q_unmask = torch.clamp(dsigma_b / esigm1, min=1e-8, max=1.0)   # [B, L]
        q_mask   = torch.clamp(1.0 - q_unmask, min=1e-8, max=1.0)      # [B, L]

        # Log-probability the model assigns to x_0 (the correct unmasked token).
        # batch: [B, L] integer token ids
        lp_x0   = log_probs.gather(-1, batch.unsqueeze(-1)).squeeze(-1)       # [B, L]
        # Log-probability the model assigns to [MASK] staying masked.
        lp_mask = log_probs[..., mask_idx]                                     # [B, L]

        # KL(q || p_theta) at each masked position (in the continuous-time limit).
        # KL = q_unmask * (log q_unmask - log p_theta(x_0))
        #    + q_mask   * (log q_mask   - log p_theta([MASK]))
        kl = (q_unmask * (q_unmask.log() - lp_x0)
              + q_mask  * (q_mask.log()  - lp_mask))      # [B, L]

        # Zero out unmasked positions (they have zero KL by construction).
        kl = kl * is_masked.float()

        # Sum over sequence positions; dsigma weighting is already folded into q_unmask.
        # We return per-sequence loss (shape [B]) to match get_loss_fn's contract.
        return kl.sum(dim=-1)

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum, loss_type="score_entropy"):
    if loss_type == "kl":
        loss_fn = get_kl_loss_fn(noise, graph, train)
    elif loss_type == "score_entropy":
        loss_fn = get_loss_fn(noise, graph, train)
    else:
        raise ValueError(f"Unknown loss_type {loss_type!r}. Choose 'score_entropy' or 'kl'.")

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn