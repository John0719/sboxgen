import torch

def bijection_loss(sbox):
    s = torch.clamp(torch.round(sbox), 0, 255).to(torch.int64)
    batch = s.shape[0]

    counts = torch.zeros(batch, 256, device=s.device, dtype=torch.int32)
    counts.scatter_add_(1, s, torch.ones_like(s, dtype=torch.int32))

    target = torch.ones(batch, 256, device=s.device)
    loss = ((counts.float() - target) ** 2).sum(dim=1)  # per sample

    return loss.mean()

def differential_uniformity(sbox):
    """
    sbox: (batch, 256) integer tensor
    returns: (batch, 255) DU vector for a=1..255
    """
    device = sbox.device
    batch = sbox.shape[0]

    # x = 0..255
    x = torch.arange(256, device=device).unsqueeze(0).unsqueeze(0)  # (1,1,256)

    # a = 1..255
    a = torch.arange(1, 256, device=device).unsqueeze(1).unsqueeze(0)  # (1,255,1)

    # Compute x⊕a for all a
    xa = x ^ a  # (1,255,256)

    # Expand sbox to match shapes
    # s = sbox.unsqueeze(1)  # (batch,1,256)
    s = sbox.unsqueeze(1).expand(-1, 255, -1)  # (batch,255,256)

    # Lookup S(x) and S(x⊕a)
    s_x  = s.expand(-1, 255, -1)          # (batch,255,256)
    s_xa = s.gather(2, xa.expand(batch, -1, -1))  # (batch,255,256)

    # Compute Δy = S(x) ⊕ S(x⊕a)
    diffs = s_x ^ s_xa  # (batch,255,256)

    # Allocate histogram buffer
    counts = torch.zeros(batch, 255, 256, device=device, dtype=torch.int32)

    # Scatter-add counts
    counts.scatter_add_(2, diffs, torch.ones_like(diffs, dtype=torch.int32))

    # DU(a) = max count per row
    du = counts.max(dim=2).values  # (batch,255)

    return du

def differential_uniformity_loss(fake_sbox, real_sbox):
    fake = torch.clamp(torch.round(fake_sbox), 0, 255).to(torch.int64)
    real = torch.clamp(torch.round(real_sbox), 0, 255).to(torch.int64)

    # Compute DU vectors (batch, 255) for differences a=1..255
    du_fake = differential_uniformity(fake)  # (batch, 255)
    du_real = differential_uniformity(real)  # (batch, 255)

    # L2 distance per sample, summed over all difference values
    loss = ((du_fake - du_real) ** 2).sum(dim=1)

    # Expectation over batch
    return loss.float().mean()

def hadamard_256(device):
    H = torch.tensor([[1]], dtype=torch.float32, device=device)
    while H.shape[0] < 256:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H

def nonlinearity(sbox):
    """
    sbox: (batch, 256) int64
    returns: (batch, 8) NF vector (nonlinearity per output bit)
    """
    device = sbox.device
    batch = sbox.shape[0]

    # Convert S-box to bits: shape (batch, 256, 8)
    bits = ((sbox.unsqueeze(-1) >> torch.arange(8, device=device)) & 1).float()

    # Walsh-Hadamard transform matrix (256x256)
    H = hadamard_256(device)

    # Compute Walsh spectrum for each output bit
    # bits.transpose(1,2): (batch, 8, 256)
    # H: (256, 256)
    W = torch.matmul(bits.transpose(1, 2), H)  # (batch, 8, 256)

    # Nonlinearity per bit = max absolute Walsh coefficient over all input combinations
    # Taking max over the coefficient dimension (dim=2)
    NF = W.abs().max(dim=2).values  # (batch, 8)

    return NF

def nonlinearity_loss(fake_sbox, real_sbox):
    fake = torch.clamp(torch.round(fake_sbox), 0, 255).to(torch.int64)
    real = torch.clamp(torch.round(real_sbox), 0, 255).to(torch.int64)

    # Compute NF vectors (batch, 8) per output bit
    nf_fake = nonlinearity(fake)  # (batch, 8)
    nf_real = nonlinearity(real)  # (batch, 8)

    # L2 distance per sample, summed over all 8 output bits
    loss = ((nf_fake - nf_real) ** 2).sum(dim=1)

    # Expectation over batch
    return loss.mean()

def bijection_loss_norm(sbox):
    bij = bijection_loss(sbox)  # your optimized version
    return bij / 65280.0

def differential_uniformity_loss_norm(fake_sbox, real_sbox):
    du = differential_uniformity_loss(fake_sbox, real_sbox)  # your optimized version
    return du / 16192512.0

def nonlinearity_loss_norm(fake_sbox, real_sbox):
    nl = nonlinearity_loss(fake_sbox, real_sbox)
    return nl / 524288.0

def critic_loss(critic, real_sbox, fake_sbox, gp_lambda=10.0):
    # WGAN part
    real_score = critic(real_sbox)
    fake_score = critic(fake_sbox.detach())
    wgan_loss = fake_score.mean() - real_score.mean()

    # Gradient penalty
    alpha = torch.rand(real_sbox.size(0), 1, device=real_sbox.device)
    alpha = alpha.expand_as(real_sbox)
    interp = alpha * real_sbox + (1 - alpha) * fake_sbox.detach()
    interp.requires_grad_(True)

    interp_score = critic(interp)
    grads = torch.autograd.grad(
        outputs=interp_score,
        inputs=interp,
        grad_outputs=torch.ones_like(interp_score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gp = ((grads.view(grads.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
    return wgan_loss + gp_lambda * gp

def ste_round(x):
    return (x.round() - x).detach() + x

def generator_loss(critic, fake_sbox, real_sbox,
                   w_du=0.25, w_nf=0.05, w_bij=0.15):
    round_fake = fake_sbox # ste_round(fake_sbox)
    round_real = real_sbox # ste_round(real_sbox)

    fake_int = torch.clamp(torch.round(fake_sbox), 0, 255).to(torch.int64)
    du_fake_vec = differential_uniformity(fake_int)  # (batch, 255)

    # Hard constraint: penalize any DU > 4
    du_max = du_fake_vec.max(dim=1).values.float()  # (batch,)
    du_violation = torch.clamp(du_max - 4, min=0)  # 0 if DU≤4, else penalty
    du_constraint_loss = (du_violation ** 2).mean() * 10.0

    w_adv = 1.0 - w_bij - w_nf - w_du
    adv = -critic(fake_sbox).mean()         # WGAN generator term
    du  = differential_uniformity_loss(round_fake, round_real)      # your DU loss
    nf  = nonlinearity_loss(round_fake, round_real)      # your NF loss
    bij = bijection_loss(round_fake)          # your bijection loss

    return w_adv * adv + w_du * du + w_nf * nf + w_bij * bij + du_constraint_loss
