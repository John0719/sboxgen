import os
import sys
import time
import argparse
import random
from datetime import datetime
import glob
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def gf2_matmul(mat, byte):
    x = torch.tensor([(byte >> i) & 1 for i in range(8)], dtype=torch.uint8)
    y = mat @ x % 2
    out = 0
    for i in range(8):
        out |= int(y[i].item()) << i
    return out

def random_invertible_matrix():
    while True:
        M = torch.randint(0, 2, (8, 8), dtype=torch.uint8)
        if torch.linalg.det(M.float()) % 2 != 0:
            return M

def affine_transform_gf256(sbox):
    A = random_invertible_matrix()
    b = torch.randint(0, 2, (8,), dtype=torch.uint8)
    out = []
    for x in sbox:
        y = gf2_matmul(A, x)
        y ^= int(sum((b[i] << i) for i in range(8)))
        out.append(y & 0xFF)
    return out

def build_dataset(n=2000):
    data = []
    for _ in range(n):
        sbox = affine_transform_gf256(AES_SBOX)
        sbox_matrix = np.array(sbox).reshape(16, 16).astype(np.float32) / 255.0
        data.append(torch.tensor(sbox_matrix, dtype=torch.float32))
    return torch.stack(data).unsqueeze(1).to(device)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 4 * 4 * 1024)
        self.conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 1024, 4, 4)

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = torch.sigmoid(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.conv1(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.conv2(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.conv3(x), 0.2)
        x = self.flatten(x)
        x = self.fc(x)
        return x.view(-1)

def differential_uniformity(sbox_flat):
    device = sbox_flat.device
    batch = sbox_flat.shape[0]

    x = torch.arange(256, device=device).unsqueeze(0).unsqueeze(0)
    a = torch.arange(1, 256, device=device).unsqueeze(1).unsqueeze(0)

    xa = x ^ a
    s = sbox_flat.unsqueeze(1).expand(-1, 255, -1)
    s_x = s.expand(-1, 255, -1)
    s_xa = s.gather(2, xa.expand(batch, -1, -1))

    diffs = s_x ^ s_xa
    counts = torch.zeros(batch, 255, 256, device=device, dtype=torch.int32)
    counts.scatter_add_(2, diffs, torch.ones_like(diffs, dtype=torch.int32))

    du = counts.max(dim=2).values
    return du

def bijection_loss(sbox):
    s = torch.clamp(torch.round(sbox * 255), 0, 255).to(torch.int64)
    batch = s.shape[0]

    s_flat = s.view(batch, -1)
    counts = torch.zeros(batch, 256, device=s.device, dtype=torch.int32)
    counts.scatter_add_(1, s_flat, torch.ones_like(s_flat, dtype=torch.int32))

    target = torch.ones(batch, 256, device=s.device)
    loss = ((counts.float() - target) ** 2).sum(dim=1)
    return loss.mean()

def differential_uniformity_loss(fake_sbox, real_sbox):
    fake = torch.clamp(torch.round(fake_sbox * 255), 0, 255).to(torch.int64)
    real = torch.clamp(torch.round(real_sbox * 255), 0, 255).to(torch.int64)

    batch = fake.shape[0]
    fake_flat = fake.view(batch, -1)
    real_flat = real.view(batch, -1)

    du_fake = differential_uniformity(fake_flat)
    du_real = differential_uniformity(real_flat)

    loss = ((du_fake.float() - du_real.float()) ** 2).sum(dim=1)
    return loss.mean()

def nonlinearity_loss(fake_sbox, real_sbox):
    fake = torch.clamp(torch.round(fake_sbox * 255), 0, 255).to(torch.int64)
    real = torch.clamp(torch.round(real_sbox * 255), 0, 255).to(torch.int64)

    batch_size = fake.shape[0]
    fake_flat = fake.view(batch_size, 256)
    real_flat = real.view(batch_size, 256)

    def compute_nf(s):
        bits = ((s.unsqueeze(-1) >> torch.arange(8, device=s.device)) & 1).float()
        return bits.mean(dim=0, keepdim=True)

    nf_fake = compute_nf(fake_flat)
    nf_real = compute_nf(real_flat)

    loss = ((nf_fake - nf_real) ** 2).mean()
    return loss

def critic_loss(critic, real_sbox, fake_sbox, gp_lambda=10.0):
    real_score = critic(real_sbox)
    fake_score = critic(fake_sbox.detach())
    wgan_loss = fake_score.mean() - real_score.mean()

    alpha = torch.rand(real_sbox.size(0), 1, 1, 1, device=real_sbox.device)
    alpha = alpha.expand_as(real_sbox)
    interp = alpha * real_sbox + (1 - alpha) * fake_sbox.detach()
    interp.requires_grad_(True)

    interp_score = critic(interp)
    grads = grad(
        outputs=interp_score,
        inputs=interp,
        grad_outputs=torch.ones_like(interp_score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gp = ((grads.view(grads.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
    return wgan_loss + gp_lambda * gp

def generator_loss(critic, fake_sbox, real_sbox, w_du=0.05, w_nf=0.05, w_bij=0.1):
    w_adv = 1.0 - w_bij - w_nf - w_du
    adv = -critic(fake_sbox).mean()
    du = differential_uniformity_loss(fake_sbox, real_sbox)
    nf = nonlinearity_loss(fake_sbox, real_sbox)
    bij = bijection_loss(fake_sbox)

    return w_adv * adv + w_du * du + w_nf * nf + w_bij * bij

G = Generator().to(device)
D = Discriminator().to(device)

LR = 0.0002
EPOCHS = 5000
BATCH_SIZE = 128
CHECKPOINT_INTERVAL = EPOCHS // 10
MODEL_DIR = "model"
RESULT_DIR = "result"
LATEST_CKPT = os.path.join(MODEL_DIR, "latest_checkpoint.pth")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

def save_checkpoint(epoch, path):
    torch.save({
        "epoch": epoch,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "optG_state": opt_G.state_dict(),
        "optD_state": opt_D.state_dict()
    }, path)

def load_checkpoint(path):
    try:
        ckpt = torch.load(path, map_location=device)
        G.load_state_dict(ckpt["G_state"])
        D.load_state_dict(ckpt["D_state"])
        opt_G.load_state_dict(ckpt["optG_state"])
        opt_D.load_state_dict(ckpt["optD_state"])
        return ckpt["epoch"]
    except RuntimeError as e:
        print(f"Checkpoint incompatible (architecture changed): {e}")
        print("Starting fresh from epoch 0")
        return 0

def train(resume=True):
    if resume and os.path.exists(LATEST_CKPT):
        start_epoch = load_checkpoint(LATEST_CKPT)
    else:
        start_epoch = 0

    dataset = build_dataset()

    pbar = tqdm(range(start_epoch, EPOCHS), desc="Training", unit="epoch")
    for epoch in pbar:
        idx = torch.randint(0, dataset.size(0), (BATCH_SIZE,), device=device)
        real = dataset[idx]

        z = torch.randn(BATCH_SIZE, 256, device=device)
        fake = G(z).detach()
        loss_D = critic_loss(D, real, fake)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        z = torch.randn(BATCH_SIZE, 256, device=device)
        fake = G(z)
        loss_G = generator_loss(D, fake, real, w_du=0.05, w_nf=0.05, w_bij=0.1)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        pbar.set_postfix({
            "G_Loss": f"{loss_G.item():.2f}",
            "D_Loss": f"{loss_D.item():.2f}"
        })

        if epoch % CHECKPOINT_INTERVAL == 0 and epoch > 0:
            save_checkpoint(epoch, LATEST_CKPT)

    torch.save(G.state_dict(), os.path.join(MODEL_DIR, "generator_final.pth"))
    torch.save(D.state_dict(), os.path.join(MODEL_DIR, "discriminator_final.pth"))

def generate_sbox():
    gen_path = os.path.join(MODEL_DIR, "generator_final.pth")
    if not os.path.exists(gen_path):
        print(f"Model not found: {gen_path}")
        return

    G.load_state_dict(torch.load(gen_path, map_location=device))
    G.eval()

    with torch.no_grad():
        z = torch.randn(1, 256, device=device)
        generated = G(z)
        sbox = torch.clamp(torch.round(generated.squeeze() * 255), 0, 255).cpu().numpy().astype(int).flatten()

    sbox = sbox[:256]
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(RESULT_DIR, f"sbox_{ts}.txt")
    with open(fname, "w") as f:
        f.write(", ".join(str(v) for v in sbox))

    print(f"S-box saved to {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('choice', nargs='?', help='t=train, g=generate')
    parser.add_argument('-m', '--mode', dest='mode', help='mode: t or g')
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()

    choice = args.mode or args.choice or input("Choose (t=train, g=generate): ").strip().lower()

    if choice == "t":
        train(resume=not args.fresh)
    elif choice == "g":
        generate_sbox()
