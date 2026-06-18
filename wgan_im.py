from datetime import datetime
import os
import time

from sbox import CRITIC_ITERS, LATEST_CKPT, LR, get_dataset, load_checkpoint
from sbox_loss import differential_uniformity
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(256, 4 * 4 * 1024)
        self.bn0 = nn.BatchNorm2d(1024)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, z):
        # z: (batch, 256)
        
        # Fully connected → reshape to (batch, 1024, 4, 4)
        x = self.fc(z)
        x = x.view(-1, 1024, 4, 4)
        
        # BatchNorm and LeakyReLU
        x = self.bn0(x)
        x = F.leaky_relu(x, 0.2)

        # Upsample to 8×8
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Conv1: 1024→512, kernel 5×5, stride 1, padding 0 → output 4×4
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)

        # Upsample to 8×8
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Conv2: 512→256, kernel 5×5, stride 1, padding 0 → output 4×4
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)

        # Upsample to 8×8
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Conv3: 256→1, kernel 5×5, stride 1, padding 0 → output 4×4
        x = self.conv3(x)
        
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv1: 1 → 128
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=5,
            stride=1,
            padding=0
        )

        # Conv2: 128 → 256 (with BatchNorm)
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.bn2 = nn.BatchNorm2d(256)

        # Conv3: 256 → 512 (with BatchNorm)
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.bn3 = nn.BatchNorm2d(512)

        # After 3 conv layers, compute final spatial size:
        # Input: 16x16
        # Conv1: 16 - 5 + 1 = 12
        # Conv2: 12 - 5 + 1 = 8
        # Conv3: 8 - 5 + 1 = 4
        # Final feature map: (512, 4, 4) = 512*16 = 8192

        self.fc = nn.Linear(512 * 4 * 4, 1)  # WGAN: output is a scalar

    def forward(self, x):
        # x: (batch, 1, 16, 16)

        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = F.leaky_relu(x, 0.2)

        x = x.view(x.size(0), -1)  # flatten

        out = self.fc(x)  # no sigmoid for WGAN

        return out

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

opt_G = optim.Adam(G.parameters(), lr=LR)
opt_D = optim.Adam(D.parameters(), lr=LR)


# --- critic loss ---
def critic_loss(critic, real, fake, penalty_lambda=10.0):
    """
    real, fake shape: (batch, 1, 16, 16)
    WGAN-IM / WGAN-GP critic loss:
        L_D = E[D(real)] - E[D(fake)] - λ * E[(||∇D(y)||2 - 1)^2]
    """

    batch_size = real.size(0)

    # ----- 1. Critic outputs -----
    real_scores = critic(real)      # shape: (batch, 1) or (batch,)
    fake_scores = critic(fake)

    wasserstein = real_scores.mean() - fake_scores.mean()

    # ----- 2. Gradient penalty -----
    alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
    y = alpha * real + (1 - alpha) * fake
    y.requires_grad_(True)

    y_scores = critic(y)

    gradients = torch.autograd.grad(
        outputs=y_scores,
        inputs=y,
        grad_outputs=torch.ones_like(y_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # flatten each sample: (batch, 1*16*16)
    grad_norm = gradients.view(batch_size, -1).norm(2, dim=1)

    grad_penalty = ((grad_norm - 1) ** 2).mean()

    # ----- 3. Final critic loss -----
    loss = -(wasserstein - penalty_lambda * grad_penalty)

    return loss

# --- Adversarial loss ---
def Adv_loss(fake):
    """
    Implements:
        L_adv = -E_z [ D(G(z)) ]

    Inputs:
        G : generator model
        D : critic / discriminator model
        z : latent batch (B, latent_dim)

    Output:
        scalar loss
    """

    # Critic score on fake samples
    score = D(fake)

    # Adversarial loss (negative mean critic score)
    loss = -torch.mean(score)

    return loss

# --- Convert generator output to S-box ---
# The output shape is (B, 256)
def f_S(gen_out):
    """
    Convert WGAN-IM / WGAN-GP generator output into a valid bijective S-box.
    
    Input:
        gen_out: Tensor of shape (B, 1, 16, 16) or (B, 16, 16) or (B, 256)
                 containing continuous generator outputs.
    
    Output:
        sboxes: Tensor of shape (B, 256) containing permutations of 0..255.
    """

    B = gen_out.shape[0]

    # 1. Flatten to (B, 256)
    if gen_out.dim() == 4:      # (B, 1, 16, 16)
        flat = gen_out.view(B, -1)
    elif gen_out.dim() == 3:    # (B, 16, 16)
        flat = gen_out.view(B, -1)
    elif gen_out.dim() == 2:    # (B, 256)
        flat = gen_out
    else:
        raise ValueError("Invalid generator output shape")

    # 2. Add tiny noise to break ties (important!)
    flat = flat + 1e-6 * torch.randn_like(flat)

    # 3. Convert continuous vector → permutation via argsort
    sboxes = flat.argsort(dim=1)

    return sboxes

RESULT_DIR = "result"
def wganim_gen_sbox():
    """
    Generate a single S-box using the trained WGAN-IM generator.
    Returns a tensor of shape (256,) containing a permutation of 0..255.
    """
    generator_path = os.path.join(MODEL_DIR, "generator_final.pth")
    if not os.path.exists(generator_path):
        print(f"No saved {generator_path} found. Train first.")
        return

    G.load_state_dict(torch.load(generator_path, map_location=DEVICE))
    G.eval()

    z = torch.randn(1, 256, device=DEVICE)
    gen = G(z)
    generated_sbox = f_S(gen)

    # Create timestamp in format YYYYMMDDHHMMSS
    ts = time.strftime("%Y%m%d_%H%M%S")
    
    fname = os.path.join(RESULT_DIR, f"sbox_{ts}.txt")
    with open(fname, "w") as f:
        f.write(", ".join(str(v) for v in generated_sbox))
        # f.write("Generated S-box (16x16 hex matrix):\n")
        # matrix = [generated_sbox[i*16:(i+1)*16] for i in range(16)]
        # for row in matrix:
        #     f.write(" ".join(f"{val:02X}" for val in row) + "\n")

    print(f"S-box generated and saved to {fname}")

# --- Deduplication feature function ---
def f_b(sbox):
    """
    Compute the deduplication number of the truth table of an S-box.

    Input:
        sbox: Tensor of shape (256,) or (B, 256) containing values 0..255

    Output:
        dedup_count: scalar tensor or tensor of shape (B,) containing the number of unique rows in the truth table
    """

    if sbox.dim() == 1:
        sbox = sbox.unsqueeze(0)
    elif sbox.dim() != 2:
        raise ValueError("f_b expects input shape (256,) or (B, 256)")

    # Convert S-box outputs to 8-bit truth table (B, 256, 8)
    # bit i = (value >> i) & 1
    bits = ((sbox.unsqueeze(2) >> torch.arange(8, device=sbox.device)) & 1).float()

    # Count unique rows per example
    dedup_counts = []
    for i in range(bits.size(0)):
        unique_rows = torch.unique(bits[i], dim=0)
        dedup_counts.append(unique_rows.shape[0])

    return torch.tensor(dedup_counts, device=sbox.device, dtype=torch.float32).squeeze(0)

# -- Bijection loss ---
def Bij_loss(fake, N=256):
    """
    Implements:
        L_b = E_z [ || f_b(f_S(G(z))) - N ||^2 ]

    Inputs:
        G   : generator model
        f_S : function that converts generator output → S-box
        f_b : bijection-related feature function
        z   : latent batch (B, latent_dim)
        N   : target tensor (same shape as f_b output)

    Output:
        scalar loss
    """

    # Convert generator output → S-box (permutation or soft)
    S = f_S(fake)

    # Apply bijection feature function
    Fb = f_b(S)

    # 4. Compute squared L2 loss
    loss = torch.mean((Fb - N) ** 2)

    return loss

# --- Differential Uniformity Loss ---
def DU_loss(real, fake):
    """
    Implements:
        L_DU = E[ || DU(f_S(G(z))) - DU(f_S(x)) ||^2 ]

    real, fake shape: (batch, 1, 16, 16)
    DU: a neural network
    f_S: shared feature extractor
    """

    # Shared feature extraction
    real_feat = f_S(real)   # (batch, F)
    fake_feat = f_S(fake)   # (batch, F)

    # DU network outputs
    real_du = differential_uniformity(real_feat).float()   # (batch, D)
    fake_du = differential_uniformity(fake_feat).float()   # (batch, D)

    # Squared L2 distance
    loss = ((fake_du - real_du) ** 2).mean()

    return loss

def NF(sboxes):
    """
    Compute the nonlinearity of a batch of 8-bit S-boxes.

    Input:
        sboxes: tensor of shape (B, 256) with values 0..255

    Output:
        nonlinearity: tensor of shape (B,) with NL for each S-box
    """

    device = sboxes.device
    B = sboxes.shape[0]

    # -----------------------------
    # 1. Convert S-box outputs to bit truth table (B, 256, 8)
    # -----------------------------
    out_bits = ((sboxes.unsqueeze(-1) >> torch.arange(8, device=device)) & 1).to(torch.int64)
    # out_bits[b, x, i] = i-th bit of S-box[b][x]

    # -----------------------------
    # 2. Precompute input bits (256, 8)
    # -----------------------------
    x = torch.arange(256, device=device)
    x_bits = ((x.unsqueeze(-1) >> torch.arange(8, device=device)) & 1).to(torch.float32)

    # -----------------------------
    # 3. Precompute all masks a (256, 8)
    # -----------------------------
    a = torch.arange(256, device=device)
    a_bits = ((a.unsqueeze(-1) >> torch.arange(8, device=device)) & 1).to(torch.float32)

    # -----------------------------
    # 4. Compute a·x mod 2 for all pairs (256, 256)
    # -----------------------------
    ax = (a_bits.unsqueeze(1) @ x_bits.unsqueeze(2)).squeeze(-1).to(torch.int64) % 2
    # ax[a, x] = dot(a_bits[a], x_bits[x]) mod 2

    # -----------------------------
    # 5. Compute Walsh spectrum for each output bit
    # -----------------------------
    NL_bits = []

    for bit in range(8):
        # f_i(x) for all S-boxes: (B, 256)
        f = out_bits[:, :, bit]  # (B, 256)

        # Expand ax to (B, 256, 256)
        exponent = (f.unsqueeze(1).to(torch.int64) ^ ax.unsqueeze(0).to(torch.int64)).float()

        # Walsh transform: sum_x (-1)^(f(x) XOR a·x)
        W = torch.sum((-1.0) ** exponent, dim=2)  # (B, 256)

        max_W = torch.max(torch.abs(W), dim=1).values  # (B,)
        NL = 128 - max_W / 2
        NL_bits.append(NL)

    # -----------------------------
    # 6. S-box nonlinearity = min over 8 output bits
    # -----------------------------
    NL_final = torch.stack(NL_bits, dim=1).min(dim=1).values  # (B,)

    return NL_final

# --- Nonlinearity Loss ---
def NF_loss(real, fake):
    """
    Implements:
        L_NF = E_{z, x} [ || NF(f_S(G(z))) - NF(f_S(x)) ||^2 ]

    Inputs:
        G      : generator model
        f_S    : function converting raw data → S-box (permutation or vector)
        NF     : normalization / feature-extraction function
        z      : latent batch (B, latent_dim)
        real_x : real S-box batch or real data batch (B, ...)

    Output:
        scalar loss
    """

    # 1. Convert fake data → S-box
    S_gen = f_S(fake)

    # 3. Convert real data → S-box
    S_real = f_S(real)

    # 4. Apply normalization / feature extractor NF
    F_gen = NF(S_gen)
    F_real = NF(S_real)

    # 5. Squared L2 loss (batch mean)
    loss = torch.mean((F_gen - F_real) ** 2)

    return loss

# --- Generator loss with additional S-box properties ---
def generator_loss(fake, real, w_adv=0.8, w_bij=0.1, w_du=0.05, w_nf=0.05):
    """
    Generator loss for WGAN-IM with additional S-box properties:
        L_G = -E[D(fake)] + w_du * DU_loss(fake, real) + w_nf * NF_loss(fake, real) + w_bij * Bij_loss(fake)
    """
    # WGAN loss
    adv_loss = Adv_loss(fake)

    # Additional S-box property losses
    du_loss = DU_loss(fake, real)
    nf_loss = NF_loss(fake, real)
    bij_loss = Bij_loss(fake)

    total_loss = w_adv * adv_loss + w_du * du_loss + w_nf * nf_loss + w_bij * bij_loss

    return total_loss

def train_wgan_im(
    z_dim=256,
    epochs=50,
    batch_size=64,
    dataset_path=None,
    save_interval=10,
    progress_callback=None,
    device=DEVICE,
    resume=False,
):
    print(f"Starting WGAN-IM training on device: {device}")
    if resume and os.path.exists(LATEST_CKPT):
        start_epoch = load_checkpoint(LATEST_CKPT)
        print(f"Resuming training from checkpoint...")
    else:
        start_epoch = 0
        if not resume and os.path.exists(LATEST_CKPT):
            print("--fresh flag set. Starting training from epoch 0 (checkpoint ignored).")
        elif not os.path.exists(LATEST_CKPT):
            print("No checkpoint found. Starting training from epoch 0.")
        else:
            print("Starting training from epoch 0.")
            
    # Prepare dataset
    dataset_path = dataset_path or os.path.join("db", "sbox_dataset.pt")
    dataset = get_dataset(dataset_path, num_samples=2000, device=device)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(start_epoch, epochs + 1):
        idx = torch.randint(0, dataset.size(0), (batch_size,), device=device)
        real = dataset[idx]
        real = real.view(real.size(0), 1, 16, 16)

        # ======================
        #  Train Critic
        # ======================
        for _ in range(CRITIC_ITERS):
            z = torch.randn(batch_size, z_dim, device=device)
            fake = G(z).detach()

            loss_D = critic_loss(D, real, fake)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        # ======================
        #  Train Generator
        # ======================
        z = torch.randn(batch_size, z_dim, device=device)
        fake = G(z)

        # WGAN loss
        adv_loss = Adv_loss(fake)

        # Additional S-box property losses
        du_loss = DU_loss(fake, real)
        nf_loss = NF_loss(fake, real)
        bij_loss = Bij_loss(fake)

        # total_loss
        loss_G = 0.8 * adv_loss + 0.05 * du_loss + 0.05 * nf_loss + 0.1 * bij_loss
        # loss_G = generator_loss(D, fake, real, w_du=0.05, w_nf=0.05, w_bij=0.1)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
        
        if progress_callback is not None:
            du_value = float(du_loss.item())
            nl_value = float(nf_loss.item())
            bij_value = float(bij_loss.item())
            progress_callback(epoch, loss_G.item(), loss_D.item(), du_value, nl_value, bij_value)

        if epoch % (epochs // 10) == 0 or epoch == epochs:
            t = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(G.state_dict(), os.path.join(MODEL_DIR, f"wgan_G_{t}_ep{epoch}.pth"))
            torch.save(D.state_dict(), os.path.join(MODEL_DIR, f"wgan_D_{t}_ep{epoch}.pth"))

    return G, D
