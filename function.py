import math

import torch
import numpy as np

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor, input tensor of any shape
        dim: int, dimension along which to apply softmax
        
    Returns:
        torch.Tensor, output tensor of the same shape as input with softmax applied along dim
    """
    max_val = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - max_val
    exp_x = torch.exp(x_shifted)

    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Args:
        Q: torch.Tensor, query tensor of shape (..., seq_len_q, d_k)
        K: torch.Tensor, key tensor of shape (..., seq_len_k, d_k)
        V: torch.Tensor, value tensor of shape (..., seq_len_v, d_v)
        mask: torch.Tensor | None, optional boolean mask of shape (..., seq_len_q, seq_len_k)
    
    Returns:
        torch.Tensor, output tensor of shape (..., seq_len_q, d_v)
    """
    d_k = Q.shape[-1]

    scaled = torch.einsum('...qd,...kd->...qk', Q, K) / (d_k ** 0.5)

    if mask is not None:
        scaled = scaled.masked_fill(mask == 0, float('-inf'))

    attention = softmax(scaled, dim=-1)
    return torch.einsum('...qk,...kv->...qv', attention, V)

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Args:
        logits: torch.Tensor, predicted logits of shape (..., vocab_size)
        targets: torch.Tensor, target indices of shape (...)
        ignore_index: int, positions with this target value are excluded from the loss
            (SFT 用: prompt 位置置 -100, 只对 response 算 loss)
    
    Returns:
        torch.Tensor: average cross-entropy loss over valid positions
    """
    vocab_dim = -1

    max_logits = torch.max(logits, dim=vocab_dim, keepdim=True)[0]
    shifted_logits = logits - max_logits

    # Compute log-softmax directly to avoid separate exp and log
    log_probs = shifted_logits - torch.logsumexp(shifted_logits, dim=vocab_dim, keepdim=True)

    if ignore_index is not None:
        safe_targets = targets.masked_fill(targets == ignore_index, 0)
    else:
        safe_targets = targets
    loss = -log_probs.gather(dim=vocab_dim, index=safe_targets.unsqueeze(vocab_dim)).squeeze(vocab_dim)
    if ignore_index is not None:
        mask = (targets != ignore_index).to(loss.dtype)
        denom = mask.sum()
        if denom == 0:
            return loss.sum() * 0.0
        return (loss * mask).sum() / denom
    return loss.mean()

def get_learning_rate(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    """
    Compute the learning rate αt according to the cosine learning rate schedule with warmup.
    
    Args:
        t (int): Current training step.
        alpha_max (float): Maximum learning rate.
        alpha_min (float): Minimum learning rate.
        T_w (int): Number of warm-up iterations.
        T_c (int): Total number of cosine annealing iterations.
    
    Returns:
        float: The learning rate αt for the current step t.
    """
    if T_w <= 0:
        raise ValueError("T_w must be positive")
    if T_c < T_w:
        raise ValueError("T_c must be greater than or equal to T_w")
    if t < T_w: # Warm-up
        return (t / T_w) * alpha_max
    elif t == T_w:
        return alpha_max
    elif T_w < t <= T_c: # Cosine annealing
        return alpha_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    else: # Post-annealing
        return alpha_min

def clip_gradient(parameters, max_norm):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    total_norm = sum(p.grad.pow(2).sum() for p in params).sqrt().item()
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            p.grad.mul_(clip_coef)
    return total_norm

def data_loader(dataset, batch_size, context_length, device):
    n = len(dataset)
    if n <= context_length:
        raise ValueError(f"dataset needs more than {context_length} tokens, got {n}")
    indices = np.random.randint(0, n - context_length, size=batch_size)

    # Gather all windows in one numpy fancy-index, then a single H2D transfer
    offsets = np.arange(context_length + 1)
    windows = dataset[indices[:, None] + offsets]  # (batch_size, context_length + 1)

    inputs = torch.from_numpy(windows[:, :-1]).to(device)
    outputs = torch.from_numpy(windows[:, 1:]).to(device)

    return inputs, outputs

def save_checkpoint(model, optimizer, iteration, out):
    state_dict = {
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'iteration' : iteration
    }
    
    torch.save(state_dict, out)

def load_checkpoint(src, model, optimizer, map_location='cpu'):
    state_dict = torch.load(src, map_location=map_location, weights_only=True)

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    iteration = state_dict['iteration']

    return iteration

def nucleus_sampling_decoder(model, tokenizer, prompt, max_length=100, temperature=1.0, p=0.9, device='cuda:0'):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    model.eval()

    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)  # Shape: (1, seq_len)
    generated_ids = input_ids.clone()
    eot_id = tokenizer.encode("<|endoftext|>")[0]

    with torch.no_grad():
        max_ctx = getattr(model, 'max_seq_len', None)
        for _ in range(max_length - input_ids.shape[1]):
            # 超出训练上下文时滑动窗口，避免 ValueError；位置编码会从 0 重算，质量会退化
            window = generated_ids
            if max_ctx is not None and generated_ids.shape[1] > max_ctx:
                window = generated_ids[:, -max_ctx:]
            logits = model(window)[:, -1, :]  # Shape: (1, vocab_size)
            logits = logits / temperature
            probs = softmax(logits, dim=-1).squeeze(0) # Shape: (vocab_size,)

            # Nucleus sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            nucleus_mask = (cumsum - sorted_probs) < p
            nucleus_indices = sorted_indices[nucleus_mask]
            nucleus_probs = sorted_probs[nucleus_mask]

            if nucleus_probs.numel() == 0:
                nucleus_probs = sorted_probs[:1]  # Take the top-1 probability
                nucleus_indices = sorted_indices[:1]  # Take the top-1 index


            nucleus_probs = nucleus_probs / (nucleus_probs.sum() + 1e-10)
            next_token = torch.multinomial(nucleus_probs, num_samples=1)
            next_token_id = nucleus_indices[next_token].item()

            # Append to generated sequence
            generated_ids = torch.cat(
                [generated_ids, torch.tensor([[next_token_id]], device=device)], dim=1
            )
            if next_token_id == eot_id:
                break
                
    return tokenizer.decode(generated_ids.squeeze(0).tolist()).replace('<|endoftext|>', '').strip()