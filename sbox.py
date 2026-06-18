import os
import random
from datetime import datetime
import sys
import argparse
import time
import subprocess
import glob
import shutil

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from tqdm import tqdm

import prepare_db

# ---------------------------------------------------------
# 0. Device
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DB_DIR = "db"
DB_PATH = os.path.join(DB_DIR, "sbox_dataset.pt")


def get_dataset(path=DB_PATH, num_samples=2000, device=device):
    if not os.path.exists(path):
        print(f"Dataset file not found at {path}. Generating new dataset...")
        prepare_db.save_dataset(path, n=num_samples, force=False)
    return prepare_db.load_dataset(path, device=device)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------
# 1. AES S-box (base dataset)
# ---------------------------------------------------------

# moved to prepare_db.py since it’s only needed there and in the dataset generation step

# ---------------------------------------------------------
# 2. Simple affine-equivalent S-box generator
# ---------------------------------------------------------

# moved to prepare_db.py since it’s only needed there and in the dataset generation step

# ---------------------------------------------------------
# 3. Generator and Discriminator (WGAN-GP)
# ---------------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # 1) Fully connected layer: 256 → 512×4×4
        self.fc = nn.Linear(256, 512 * 4 * 4)
        self.bn1 = nn.BatchNorm2d(512)

        # 2) First conv layer: 512 → 256
        self.conv1 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.bn2 = nn.BatchNorm2d(256)

        # 3) Second conv layer: 256 → 1
        self.conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=2
        )
        
    def forward(self, z):
        # z: (batch, 256)

        # Fully connected → reshape to (batch, 512, 4, 4)
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)

        # Batch normalization + LeakyReLU
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)

        # Upsample to 8×8
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Conv1: 512→256, kernel 5×5, stride 1, padding 2 → output 8×8
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)

        # Upsample to 8×8 again
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Conv2: 256→1, kernel 5×5, stride 1, padding 2 → output 16×16
        x = self.conv2(x)

        # Final output: reshape to (batch, 1, 16, 16)
        # Because conv2 output is (batch, 1, 4, 4)
        # but we upsampled twice → final size is already 16×16
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
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)

        x = x.view(x.size(0), -1)  # flatten

        out = self.fc(x)  # no sigmoid for WGAN

        return out

G = Generator().to(device)
D = Discriminator().to(device)

# ---------------------------------------------------------
# 4. WGP-IM Losses
# ---------------------------------------------------------

from sbox_loss import *

# ---------------------------------------------------------
# 5. Gradient penalty
# ---------------------------------------------------------
# def gradient_penalty(real, fake):
#     batch_size = real.size(0)
#     alpha = torch.rand(batch_size, 1, device=device)
#     interpolates = alpha * real + (1 - alpha) * fake
#     interpolates.requires_grad_(True)
#     d_interpolates = D(interpolates)
#     gradients = grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=torch.ones_like(d_interpolates),
#         create_graph=True,
#         retain_graph=True
#     )[0]
#     return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def gradient_penalty(critic, real, fake, device, lambda_gp=10.0):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    critic_interpolated = critic(interpolated)

    grads = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)
    gp = lambda_gp * ((grad_norm - 1) ** 2).mean()
    return gp

# ---------------------------------------------------------
# 6. Checkpoint helpers
# ---------------------------------------------------------
def save_checkpoint(epoch, path):
    torch.save({
        "epoch": epoch,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "optG_state": opt_G.state_dict(),
        "optD_state": opt_D.state_dict()
    }, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(path):
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    opt_G.load_state_dict(ckpt["optG_state"])
    opt_D.load_state_dict(ckpt["optD_state"])
    print(f"Resumed from checkpoint at epoch {ckpt['epoch']}")
    return ckpt["epoch"]

# ---------------------------------------------------------
# 7. Training Loop
# ---------------------------------------------------------
Z_DIM = 256
LR = 1e-4
EPOCHS = 50
BATCH_SIZE = 64
CRITIC_ITERS = 5
LAMDA_GP = 10.0
CHECKPOINT_INTERVAL = EPOCHS // 10
MODEL_DIR = "model"
RESULT_DIR = "result"
LATEST_CKPT = os.path.join(MODEL_DIR, "latest_checkpoint.pth")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

opt_G = optim.Adam(G.parameters(), lr=LR)
opt_D = optim.Adam(D.parameters(), lr=LR)

def train(resume=True, progress_callback=None):
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
    dataset = get_dataset(DB_PATH, num_samples=2000, device=device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Check the DU of databset code snippet
    du_real = differential_uniformity(dataset[0:1].to(torch.int64))
    print("Real sample DU max:", du_real.max().item())

    pbar = tqdm(range(start_epoch, EPOCHS), desc="Training", unit="epoch")
    for epoch in pbar:
        idx = torch.randint(0, dataset.size(0), (BATCH_SIZE,), device=device)
        real = dataset[idx]
        real = real.view(real.size(0), 1, 16, 16)

        # ======================
        #  Train Critic
        # ======================
        for _ in range(CRITIC_ITERS):
            z = torch.randn(BATCH_SIZE, Z_DIM, device=device)
            fake = G(z).detach()

            D_real = D(real)
            D_fake = D(fake)

            gp = gradient_penalty(D, real, fake, device, lambda_gp=LAMDA_GP)

            loss_D = D_fake.mean() - D_real.mean() + gp
            # loss_D = critic_loss(D, real, fake)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        # ======================
        #  Train Generator
        # ======================
        z = torch.randn(BATCH_SIZE, Z_DIM, device=device)
        fake = G(z)
        D_fake = D(fake)
        loss_G = -D_fake.mean()
        # loss_G = generator_loss(D, fake, real, w_du=0.05, w_nf=0.05, w_bij=0.1)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        real = real.view(real.size(0), -1)
        fake = fake.view(fake.size(0), -1)
        # Monitor raw metrics for plotting and diagnostics
        # compute metrics and detach tensors before converting to Python floats
        du_res = differential_uniformity_loss(fake, real)
        if isinstance(du_res, torch.Tensor):
            du_value = float(du_res.detach().cpu().item())
        else:
            du_value = float(du_res)

        nl_res = nonlinearity_loss(fake, real)
        if isinstance(nl_res, torch.Tensor):
            nl_value = float(nl_res.detach().cpu().item())
        else:
            nl_value = float(nl_res)

        bij_value = float(bijection_loss(fake))

        fake_int = torch.clamp(torch.round(fake), 0, 255).to(torch.int64)
        du_fake_vec = differential_uniformity(fake_int[:1])
        
        # Update progress bar with loss information
        pbar.set_postfix({
            "G_Loss": f"{loss_G.item():.2f}",
            "D_Loss": f"{loss_D.item():.2f}",
            "BIJ": f"{bij_value:.2f}",
            "DU": f"{du_value:.2f}",
            "NL": f"{nl_value:.2f}"
        })

        if progress_callback:
            progress_callback(
                epoch,
                loss_G.item(),
                loss_D.item(),
                du_value,
                nl_value,
                bij_value
            )

        if epoch % CHECKPOINT_INTERVAL == 0 and epoch > 0:
            ckpt_name = os.path.join(MODEL_DIR, f"checkpoint_{timestamp()}_epoch{epoch}.pth")
            print(f"\n[DEBUG] epoch {epoch} DU(fake[0]) max =", du_fake_vec.max().item())
            save_checkpoint(epoch, ckpt_name)
            save_checkpoint(epoch, LATEST_CKPT)

    # Save final models
    generator_path = os.path.join(MODEL_DIR, "generator_final.pth")
    discriminator_path = os.path.join(MODEL_DIR, "discriminator_final.pth")
    torch.save(G.state_dict(), generator_path)
    torch.save(D.state_dict(), discriminator_path)
    print(f"Training complete. Models saved: {generator_path}, {discriminator_path}")
    return {
        "epochs": EPOCHS,
        "last_G_loss": loss_G.item(),
        "last_D_loss": loss_D.item(),
        "last_DU": du_value,
        "last_NL": nl_value,
        "last_bij": bij_value
    }

# ---------------------------------------------------------
# 8. Result directory cleanup
# ---------------------------------------------------------
def clean_result_dir():
    if not os.path.isdir(RESULT_DIR):
        print(f"No result directory found at {RESULT_DIR}.")
        return

    for entry in os.listdir(RESULT_DIR):
        path = os.path.join(RESULT_DIR, entry)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception as e:
            print(f"Failed to remove {path}: {e}")
    print(f"Cleaned result directory: {RESULT_DIR}")


def clean_model_dir():
    if not os.path.isdir(MODEL_DIR):
        print(f"No model directory found at {MODEL_DIR}.")
        return

    for entry in os.listdir(MODEL_DIR):
        path = os.path.join(MODEL_DIR, entry)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception as e:
            print(f"Failed to remove {path}: {e}")
    print(f"Cleaned model directory: {MODEL_DIR}")

# ---------------------------------------------------------
# 9. Export to ONNX
# ---------------------------------------------------------
def export_onnx():
    dummy = torch.randn(1, 128, device=device)
    onnx_name = os.path.join(MODEL_DIR, f"generator_{timestamp()}.onnx")
    torch.onnx.export(
        G, dummy, onnx_name,
        input_names=["noise"],
        output_names=["sbox"],
        opset_version=17
    )
    print(f"Generator exported to {onnx_name}")

# ---------------------------------------------------------
# 9. Generate and save S-box
# ---------------------------------------------------------
def generate_sbox():
    generator_path = os.path.join(MODEL_DIR, "generator_final.pth")
    if not os.path.exists(generator_path):
        print(f"No saved {generator_path} found. Train first.")
        return

    G.load_state_dict(torch.load(generator_path, map_location=device))
    G.eval()

    z = torch.randn(1, Z_DIM, device=device)
    generated_sbox = G(z).detach().cpu().numpy().flatten()
    # Round and clamp to valid S-box range
    generated_sbox = np.clip(np.round(generated_sbox), 0, 255).astype(int)
    
    s = torch.from_numpy(generated_sbox)
    batched_sbox = s.unsqueeze(0)  # Add batch dimension
    du = differential_uniformity(batched_sbox).float().mean()
    print(f"Generated S-box DU: {du.item():.2f}")
    
    generated_sbox = [int(v) for v in generated_sbox]

    
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

# ---------------------------------------------------------
# 10. Analyze S-box
# ---------------------------------------------------------
def analyze_sbox(sbox_file=None):
    # Use provided file if available, otherwise find the latest generated S-box file
    if sbox_file:
        if not os.path.exists(sbox_file):
            print(f"S-box file not found: {sbox_file}")
            return
        latest_sbox = sbox_file
    else:
        sbox_files = glob.glob(os.path.join(RESULT_DIR, "sbox_*.txt"))
        if not sbox_files:
            print(f"No S-box files found in {RESULT_DIR}. Generate one first.")
            return
        latest_sbox = max(sbox_files, key=os.path.getctime)

    print(f"Analyzing: {latest_sbox}")

    # Create output directory for analysis results
    sbox_basename = os.path.splitext(os.path.basename(latest_sbox))[0]
    output_dir = os.path.join(RESULT_DIR, f"{sbox_basename}_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Run the analyzer script
    analyzer_script = os.path.join("analyzer", "SRC", "Sbox_analyzer.py")
    if not os.path.exists(analyzer_script):
        print(f"Analyzer script not found at {analyzer_script}")
        return

    try:
        cmd = [sys.executable, analyzer_script, "--file", latest_sbox, "--output-dir", output_dir]
        subprocess.run(cmd, check=True)
        print(f"\n✅ Analysis results saved to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error running analyzer: {e}")

