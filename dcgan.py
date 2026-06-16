"""
DCGAN implementation for 16x16 S-box representations (Section 4.1)
- Provides `GeneratorDCGAN` and `DiscriminatorDCGAN` classes
- `train_dcgan()` trains with a standard DCGAN adversarial (BCE) loss
- Uses the dataset saved/loaded via `prepare_db` (expects 1x16x16 tensors scaled 0..1)
"""

import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import prepare_db

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


class GeneratorDCGAN(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        neg_slope = 0.2  # LeakyReLU slope

        self.net = nn.Sequential(
            # Layer 1: (latent_dim, 1, 1) -> (64, 4, 4)
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Layer 2: (64, 4, 4) -> (32, 8, 8)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Layer 3: (32, 8, 8) -> (16, 16, 16)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(neg_slope, inplace=True),

            # Layer 4 (final): keep 16×16, output 1 channel
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class DiscriminatorDCGAN(nn.Module):
    def __init__(self):
        super().__init__()

        neg_slope = 0.2  # LeakyReLU slope

        self.features = nn.Sequential(
            # Input: (batch, 1, 16, 16) → (batch, 16, 6, 6)
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(neg_slope, inplace=True),

            # (batch, 16, 6, 6) → (batch, 32, 1, 1)
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(neg_slope, inplace=True),

            # (batch, 32, 1, 1) → (batch, 64, 1, 1)
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(neg_slope, inplace=True),

            # (batch, 64, 1, 1) → (batch, 128, 1, 1)
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

        # Final classifier: 128 → 2 (real/fake)
        self.classifier = nn.Linear(128, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features
        out = self.features(x)          # shape: (batch, 128, 1, 1)
        out = out.view(out.size(0), -1) # flatten to (batch, 128)

        # Classify into real/fake
        logits = self.classifier(out)   # (batch, 2)

        # Softmax probability distribution
        probs = F.softmax(logits, dim=1)

        return probs


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def train_dcgan(
    z_dim=128,
    epochs=50,
    batch_size=64,
    lr=2e-4,
    beta1=0.5,
    dataset_path=None,
    save_interval=10,
    device=DEVICE,
):
    dataset_path = dataset_path or os.path.join("db", "sbox_dataset.pt")

    # Load dataset (prepare_db returns cpu tensor; we'll move batches to device)
    dataset = prepare_db.load_dataset(dataset_path, device="cpu")
    ds = TensorDataset(dataset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    G = GeneratorDCGAN(z_dim=z_dim).to(device)
    D = DiscriminatorDCGAN().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    optD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_z = torch.randn(16, z_dim, device=device)

    real_label = 1.0
    fake_label = 0.0

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        for (real_batch,) in loader:
            real = real_batch.to(device)
            bsz = real.size(0)

            # Train Discriminator
            D.zero_grad()
            labels = torch.full((bsz,), real_label, device=device)
            out_real = D(real)
            loss_real = criterion(out_real, labels)

            z = torch.randn(bsz, z_dim, device=device)
            fake = G(z).detach()
            labels.fill_(fake_label)
            out_fake = D(fake)
            loss_fake = criterion(out_fake, labels)

            lossD = loss_real + loss_fake
            lossD.backward()
            optD.step()

            # Train Generator
            G.zero_grad()
            labels.fill_(real_label)  # try to fool discriminator
            z = torch.randn(bsz, z_dim, device=device)
            gen = G(z)
            out = D(gen)
            lossG = criterion(out, labels)
            lossG.backward()
            optG.step()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{epochs} — lossD: {lossD.item():.4f}, lossG: {lossG.item():.4f}, time: {elapsed:.1f}s")

        if epoch % save_interval == 0 or epoch == epochs:
            t = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(G.state_dict(), os.path.join(MODEL_DIR, f"dcgan_G_{t}_ep{epoch}.pth"))
            torch.save(D.state_dict(), os.path.join(MODEL_DIR, f"dcgan_D_{t}_ep{epoch}.pth"))

    return G, D


def generate_samples(G, num_samples=16, z_dim=128, device=DEVICE):
    G.eval()
    z = torch.randn(num_samples, z_dim, device=device)
    with torch.no_grad():
        samples = G(z).cpu()
    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DCGAN for S-box generation")
    parser.add_argument("--db", default=os.path.join("db", "sbox_dataset.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--zdim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-interval", type=int, default=10)
    args = parser.parse_args()

    train_dcgan(z_dim=args.zdim, epochs=args.epochs, batch_size=args.batch, lr=args.lr, dataset_path=args.db, save_interval=args.save_interval)
