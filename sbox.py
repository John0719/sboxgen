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
import torch.optim as optim
from torch.autograd import grad
from tqdm import tqdm

# ---------------------------------------------------------
# 0. Device
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------
# 1. AES S-box (base dataset)
# ---------------------------------------------------------
AES_SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
]

# ---------------------------------------------------------
# 2. Simple affine-equivalent S-box generator
# ---------------------------------------------------------
def gf2_matmul(mat, byte):
    """
    mat: (8,8) binary matrix (0/1)
    byte: integer 0..255
    returns: integer 0..255
    """
    x = torch.tensor([(byte >> i) & 1 for i in range(8)], dtype=torch.uint8)
    y = mat @ x % 2
    out = 0
    for i in range(8):
        out |= int(y[i].item()) << i
    return out

def random_invertible_matrix():
    while True:
        M = torch.randint(0, 2, (8, 8), dtype=torch.uint8)
        if torch.linalg.det(M.float()) % 2 != 0:  # invertible mod 2
            return M

def affine_transform_gf256(sbox):
    A = random_invertible_matrix()          # 8x8 binary matrix
    b = torch.randint(0, 2, (8,), dtype=torch.uint8)  # 8-bit vector

    out = []
    for x in sbox:
        y = gf2_matmul(A, x)                # A * x
        y ^= int(sum((b[i] << i) for i in range(8)))  # XOR b
        out.append(y & 0xFF)

    return out

def affine_transform(sbox):
    a = random.randint(1, 255)
    b = random.randint(0, 255)
    return [(a * x ^ b) & 0xFF for x in sbox]

def build_dataset(n=2000):
    data = []
    for _ in range(n):
        data.append(torch.tensor(affine_transform_gf256(AES_SBOX), dtype=torch.float32))
    return torch.stack(data).to(device)

# ---------------------------------------------------------
# 3. Generator and Discriminator (WGAN-GP)
# ---------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=128, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, 256),
        )

    def forward(self, z):
        # z: (batch, z_dim)
        x = self.net(z)              # (batch, 256), real-valued
        x = torch.sigmoid(x) * 255.0 # map to [0,255]
        return x                     # you’ll round/clamp in the crypto losses

class Discriminator(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sbox):
        # sbox: (batch, 256), real or fake
        return self.net(sbox).view(-1)  # (batch,)

G = Generator().to(device)
D = Discriminator().to(device)

# ---------------------------------------------------------
# 4. WGP-IM Losses
# ---------------------------------------------------------

from sbox_loss import *

# ---------------------------------------------------------
# 5. Gradient penalty
# ---------------------------------------------------------
def gradient_penalty(real, fake):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

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
LR = 1e-4
EPOCHS = 200
BATCH_SIZE = 32
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
    dataset = build_dataset()

    # Check the DU of databset code snippet
    du_real = differential_uniformity(dataset[0:1].to(torch.int64))
    print("Real sample DU max:", du_real.max().item())

    pbar = tqdm(range(start_epoch, EPOCHS), desc="Training", unit="epoch")
    for epoch in pbar:
        idx = torch.randint(0, dataset.size(0), (BATCH_SIZE,), device=device)
        real = dataset[idx]

        # Train Discriminator
        z = torch.randn(BATCH_SIZE, 128, device=device)
        fake = G(z).detach()
        # loss_D = -(D(real).mean() - D(fake).mean()) + 10 * gradient_penalty(real, fake)
        loss_D = critic_loss(D, real, fake)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator
        z = torch.randn(BATCH_SIZE, 128, device=device)
        fake = G(z)

        loss_G = generator_loss(D, fake, real, w_du=0.05, w_nf=0.05, w_bij=0.1)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

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

    z = torch.randn(1, 128, device=device)
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

# ---------------------------------------------------------
# 11. Interactive mode
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sboxgen options")
    parser.add_argument('choice', nargs='?', help='T=train, G=generate, O=export, A=analyze')
    parser.add_argument('target', nargs='?', help='For analyze mode: path to S-box file')
    parser.add_argument('-m', '--mode', dest='mode', help='mode: t (train), g (generate), o (export), a (analyze)')
    parser.add_argument('-f', '--file', dest='file', help='S-box file to analyze when mode is a')
    parser.add_argument('--fresh', action='store_true', help='Start training from beginning (ignore checkpoint)')
    parser.add_argument('-cr', '--clean-results', action='store_true', help='Remove all files and subdirectories inside the result directory')
    parser.add_argument('-cm', '--clean-models', action='store_true', help='Remove all files and subdirectories inside the model directory')
    args = parser.parse_args()

    if args.clean_results:
        clean_result_dir()
        sys.exit(0)

    if args.clean_models:
        clean_model_dir()
        sys.exit(0)

    choice = None
    if args.mode:
        choice = args.mode
    elif args.choice:
        choice = args.choice
    else:
        print("Press T to train, G to generate S-box from saved model, O to export ONNX, A to analyze S-box, C to clean results, M to clean models:")
        choice = input("Your choice: ").strip()

    choice = choice.strip().lower() if choice else ""

    if choice == "t":
        resume = not args.fresh
        train(resume=resume)
    elif choice == "g":
        generate_sbox()
    elif choice == "o":
        export_onnx()
    elif choice == "a":
        sbox_file = args.file or args.target
        analyze_sbox(sbox_file)
    elif choice == "c":
        clean_result_dir()
    elif choice == "m":
        clean_model_dir()
    else:
        print("Unknown choice.")
