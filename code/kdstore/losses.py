"""The censored distillation objective.

The artifact holds K probabilities and their total mass. It does not hold the
shape of the tail, so the divergence has to be taken over a partition the
consumer can actually see: the K stored entries, plus one bucket for
everything else.

    L_KD = sum_{k<=K} p_k log(p_k / q_k) + (1-m) log((1-m) / (1-s))

with m the teacher's stored mass and s the student's mass on the same K
identities. Note what this does *not* do: it never renormalises the K
probabilities to sum to one. Renormalising is the standard shortcut and it
inflates the kept modes, which leaves the student over-confident in them;
carrying the residual as its own bucket is the correction Anshumann et al.
call a ghost token, and under it the gradient on the stored entries matches
the full-distribution gradient. What the form cannot fix is the missing tail
shape, and we measure that instead of assume it away.
"""
import torch

TAIL_EPS = 1e-6


def censored_kd_loss(logits, gold, topk_ids, topk_probs, gate, lam=1.0,
                     temperature=1.0):
    """Per-position loss for one chunk of positions.

    logits       [m, V] float32
    gold         [m]    gold token ids
    topk_ids     [m, K] teacher's stored identities
    topk_probs   [m, K] teacher's stored probabilities
    gate         [m]    bool, which positions the gate kept

    Gated positions get lam * KD + (1 - lam) * CE; the rest get cross-entropy
    on the gold token. Returns (summed loss, summed KD term, position count).

    On temperature: the store has the tail's total mass but not the individual
    probabilities inside it, and (sum x_i)^a is not sum (x_i^a). Tempering the
    stored view therefore has to treat the tail as one lump, which understates
    the tempered tail mass, by 8.4x at T=2 on our teacher. That is why the
    paper runs at T=1 and reports the discrepancy instead of distilling at
    higher temperatures and calling the result a temperature effect.
    """
    log_z = torch.logsumexp(logits, -1)
    ce = log_z - logits.gather(-1, gold[:, None]).squeeze(-1)

    if topk_probs is None or not gate.any():
        zero = torch.zeros((), device=logits.device)
        return ce.sum(), zero, ce.numel()

    mass = topk_probs.sum(-1).clamp(max=1.0 - TAIL_EPS)

    if temperature != 1.0:
        power = 1.0 / temperature
        tempered = topk_probs.clamp_min(1e-12) ** power
        tail = (1.0 - mass).clamp_min(TAIL_EPS) ** power
        topk_probs = tempered / (tempered.sum(-1) + tail)[:, None]
        mass = topk_probs.sum(-1).clamp(max=1.0 - TAIL_EPS)
        logits = logits / temperature
        log_z = torch.logsumexp(logits, -1)

    # Filter the logits by the gate *before* gathering. `gather` tolerates an
    # index smaller than the input along non-gathered dimensions, so writing
    # logits.gather(-1, topk_ids[gate]) quietly reads the first few rows of
    # logits instead of the gated ones. It is correct only when the gate is
    # all-True, which is to say it is wrong exactly on the arms under test.
    log_q = logits[gate].gather(-1, topk_ids[gate]) - log_z[gate][:, None]
    p = topk_probs[gate].clamp_min(0)

    head = (p * (torch.log(p.clamp_min(1e-12)) - log_q)).sum(-1)
    student_mass = log_q.exp().sum(-1).clamp(max=1.0 - TAIL_EPS)
    teacher_mass = mass[gate]
    tail = (1 - teacher_mass) * (
        torch.log((1 - teacher_mass).clamp_min(1e-12))
        - torch.log((1 - student_mass).clamp_min(TAIL_EPS)))

    kd = torch.zeros_like(ce)
    # The KD gradient shrinks as 1/T^2, so scale back up to keep the effective
    # step size independent of temperature (Hinton's correction).
    kd[gate] = (head + tail) * (temperature * temperature)

    per_position = torch.where(gate, lam * kd + (1 - lam) * ce, ce)
    return per_position.sum(), kd[gate].detach().sum(), ce.numel()
