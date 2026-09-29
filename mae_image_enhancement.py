# ============================================================
# SELF-SUPERVISED REPRESENTATION LEARNING FOR IMAGE ENHANCEMENT
# Masked Autoencoder-style Image Enhancement on CIFAR-10
# Final Course Report Version
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
OUTPUT_DIR = "report_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 10                  # Use 10 for report; 15–20 gives better results if time allows
BATCH_SIZE = 64
LR = 1e-3
NOISE_STD = 0.15
PATCH_SIZE = 4
MASK_RATIO = 0.75
EVAL_IMAGES = 300            # Number of test images used for quantitative evaluation
FIG_DPI = 300

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ============================================================
# DATASET
# ============================================================

print("=" * 70)
print("SELF-SUPERVISED REPRESENTATION LEARNING FOR IMAGE ENHANCEMENT")
print("=" * 70)
print(f"Device: {device}")
print("Loading CIFAR-10 dataset...")

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True if device == "cuda" else False
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if device == "cuda" else False
)

class_names = train_dataset.classes

print(f"Training images: {len(train_dataset)}")
print(f"Testing images : {len(test_dataset)}")


# ============================================================
# IMAGE CORRUPTION AND MASKING
# ============================================================

def add_gaussian_noise(x, std=NOISE_STD):
    """
    Adds Gaussian noise to an image tensor and clamps values to [0, 1].
    """
    noisy = x + torch.randn_like(x) * std
    return torch.clamp(noisy, 0.0, 1.0)


def mask_patches(x, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO):
    """
    MAE-style random patch masking.

    Returns:
        masked_image : image with randomly hidden patches
        visible_mask : 1 for visible pixels, 0 for hidden pixels
        hidden_mask  : 1 for hidden pixels, 0 for visible pixels
    """
    B, C, H, W = x.shape

    assert H % patch_size == 0 and W % patch_size == 0, \
        "Image height and width must be divisible by patch size."

    ph = H // patch_size
    pw = W // patch_size
    total_patches = ph * pw

    random_scores = torch.rand(B, total_patches, device=x.device)
    num_visible = int(total_patches * (1.0 - mask_ratio))

    visible_indices = random_scores.argsort(dim=1)[:, :num_visible]
    patch_mask = torch.zeros(B, total_patches, device=x.device)
    patch_mask.scatter_(1, visible_indices, 1.0)

    patch_mask = patch_mask.view(B, 1, ph, pw)
    visible_mask = patch_mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    hidden_mask = 1.0 - visible_mask

    masked_image = x * visible_mask

    return masked_image, visible_mask, hidden_mask


# ============================================================
# DIFFERENTIABLE SSIM LOSS
# ============================================================

class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss.
    The loss is 1 - SSIM, so lower values indicate better structural similarity.
    """
    def __init__(self, channels=3, window_size=11, sigma=1.5):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.sigma = sigma

        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        kernel_2d = torch.outer(g, g)
        kernel_2d = kernel_2d.view(1, 1, window_size, window_size)
        kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)

        self.register_buffer("window", kernel_2d)

    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, self.window, padding=pad, groups=self.channels)
        mu_target = F.conv2d(target, self.window, padding=pad, groups=self.channels)

        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred * pred, self.window, padding=pad, groups=self.channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, self.window, padding=pad, groups=self.channels) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, self.window, padding=pad, groups=self.channels) - mu_cross

        ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        return 1.0 - ssim_map.mean()


class ReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss for self-supervised image enhancement.

    The loss includes:
    1. Full-image MSE loss
    2. Masked-region MSE loss
    3. SSIM structural loss
    """
    def __init__(self, full_weight=0.45, masked_weight=0.35, ssim_weight=0.20):
        super().__init__()
        self.full_weight = full_weight
        self.masked_weight = masked_weight
        self.ssim_weight = ssim_weight
        self.ssim_loss = SSIMLoss()

    def forward(self, pred, target, hidden_mask):
        full_mse = F.mse_loss(pred, target)

        masked_error = (pred - target) ** 2
        masked_mse = (masked_error * hidden_mask).sum() / (hidden_mask.sum() * pred.shape[1] + 1e-8)

        structural_loss = self.ssim_loss(pred, target)

        total_loss = (
            self.full_weight * full_mse +
            self.masked_weight * masked_mse +
            self.ssim_weight * structural_loss
        )

        return total_loss, full_mse.item(), masked_mse.item(), structural_loss.item()


# ============================================================
# MODEL
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()

        stride = 2 if downsample else 1

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MAEEnhancer(nn.Module):
    """
    CNN-based Masked Autoencoder for image enhancement.

    Encoder:
        Learns visual representations from noisy and partially masked images.

    Bottleneck:
        Stores compact learned representation.

    Decoder:
        Reconstructs the clean enhanced image.
    """
    def __init__(self):
        super().__init__()

        self.encoder1 = ConvBlock(3, 64, downsample=False)
        self.encoder2 = ConvBlock(64, 128, downsample=True)
        self.encoder3 = ConvBlock(128, 256, downsample=True)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ConvBlock(128, 128, downsample=False)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ConvBlock(64, 64, downsample=False)
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        z = self.bottleneck(x)
        x = self.decoder2(z)
        x = self.decoder1(x)
        out = self.output_layer(x)
        return out

    def encode(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        z = self.bottleneck(x)
        z = F.adaptive_avg_pool2d(z, output_size=1)
        return z.flatten(1)


model = MAEEnhancer().to(device)
criterion = ReconstructionLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)
print(f"Model: CNN-based Masked Autoencoder")
print(f"Trainable parameters: {total_params:,}")
print(f"Noise level: sigma = {NOISE_STD}")
print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}")
print(f"Mask ratio: {MASK_RATIO * 100:.0f}%")
print("=" * 70)


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def tensor_to_numpy_image(x):
    return x.detach().cpu().permute(1, 2, 0).numpy()


def compute_single_metrics(clean, test_img):
    clean_np = np.clip(tensor_to_numpy_image(clean), 0, 1)
    test_np = np.clip(tensor_to_numpy_image(test_img), 0, 1)

    psnr_value = psnr(clean_np, test_np, data_range=1.0)
    ssim_value = ssim(clean_np, test_np, data_range=1.0, channel_axis=2)

    return psnr_value, ssim_value


# ============================================================
# FIGURE 1: MODEL WORKFLOW / ARCHITECTURE
# ============================================================

def save_architecture_figure():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis("off")

    boxes = [
        ("Clean CIFAR-10\nImage", 0.05),
        ("Gaussian Noise\nCorruption", 0.22),
        ("75% Patch\nMasking", 0.39),
        ("Encoder\nRepresentation\nLearning", 0.56),
        ("Decoder\nReconstruction", 0.73),
        ("Enhanced\nOutput", 0.90)
    ]

    for text, x in boxes:
        ax.text(
            x, 0.55, text,
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="black", linewidth=1.5)
        )

    for i in range(len(boxes) - 1):
        x1 = boxes[i][1] + 0.065
        x2 = boxes[i + 1][1] - 0.065
        ax.annotate(
            "",
            xy=(x2, 0.55),
            xytext=(x1, 0.55),
            arrowprops=dict(arrowstyle="->", linewidth=1.8)
        )

    ax.text(
        0.5, 0.18,
        "Self-supervised learning is performed through reconstruction of hidden image patches without using class labels.",
        ha="center", va="center", fontsize=11
    )

    fig.suptitle(
        "Figure 1. Proposed Self-Supervised MAE Framework for Image Enhancement",
        fontsize=15, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_mae_workflow_architecture.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


# ============================================================
# FIGURE 2: PATCH MASKING DEMONSTRATION
# ============================================================

def save_masking_demo():
    model.eval()
    imgs, _ = next(iter(test_loader))
    clean = imgs[:1].to(device)
    noisy = add_gaussian_noise(clean, NOISE_STD)
    masked, visible_mask, hidden_mask = mask_patches(noisy)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

    images = [
        clean[0],
        noisy[0],
        masked[0],
        hidden_mask[0].repeat(3, 1, 1)
    ]

    titles = [
        "Original Clean Image",
        "Noisy Image",
        "Masked Noisy Image",
        "Hidden Patch Map"
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.clip(tensor_to_numpy_image(img), 0, 1))
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "Figure 2. MAE Patch Masking Strategy for Self-Supervised Reconstruction",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_patch_masking_demo.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


# ============================================================
# TRAINING
# ============================================================

save_architecture_figure()
save_masking_demo()

print("Starting training...")

all_total_losses = []
all_full_mse = []
all_masked_mse = []
all_ssim_loss = []
epoch_losses = []

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    running_batches = 0

    for step, (imgs, _) in enumerate(train_loader):
        clean = imgs.to(device)

        noisy = add_gaussian_noise(clean, NOISE_STD)
        masked, visible_mask, hidden_mask = mask_patches(noisy)

        pred = model(masked)

        loss, full_mse, masked_mse, structural_loss = criterion(pred, clean, hidden_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_total_losses.append(loss.item())
        all_full_mse.append(full_mse)
        all_masked_mse.append(masked_mse)
        all_ssim_loss.append(structural_loss)

        running_loss += loss.item()
        running_batches += 1

        if step % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Step [{step:04d}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_epoch_loss = running_loss / running_batches
    epoch_losses.append(avg_epoch_loss)
    scheduler.step()

    print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

torch.save(model.state_dict(), f"{OUTPUT_DIR}/mae_enhancer_model.pth")
print("Training completed.")
print(f"Model saved to: {OUTPUT_DIR}/mae_enhancer_model.pth")


# ============================================================
# FIGURE 3: TRAINING CURVES
# ============================================================

def moving_average(values, window=50):
    values = np.array(values)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(all_total_losses, linewidth=0.8, alpha=0.45, label="Iteration Loss")
    ma = moving_average(all_total_losses, window=50)
    axes[0].plot(range(len(ma)), ma, linewidth=2.0, label="Moving Average")
    axes[0].set_title("Training Loss per Iteration", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Combined Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(1, EPOCHS + 1), epoch_losses, marker="o", linewidth=2.2)
    axes[1].set_title("Average Training Loss per Epoch", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Average Loss")
    axes[1].set_xticks(range(1, EPOCHS + 1))
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "Figure 3. Self-Supervised MAE Training Loss Curves",
        fontsize=15, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_training_loss_curves.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


save_training_curves()


# ============================================================
# EVALUATION
# ============================================================

print("Running quantitative evaluation...")

model.eval()

clean_list = []
noisy_list = []
masked_list = []
enhanced_list = []
label_list = []

psnr_noisy = []
psnr_enhanced = []
ssim_noisy = []
ssim_enhanced = []

with torch.no_grad():
    total_seen = 0

    for imgs, labels in test_loader:
        clean = imgs.to(device)
        labels = labels.to(device)

        noisy = add_gaussian_noise(clean, NOISE_STD)
        masked, _, _ = mask_patches(noisy)
        enhanced = model(masked)

        for i in range(clean.shape[0]):
            pn, sn = compute_single_metrics(clean[i], noisy[i])
            pe, se = compute_single_metrics(clean[i], enhanced[i])

            psnr_noisy.append(pn)
            psnr_enhanced.append(pe)
            ssim_noisy.append(sn)
            ssim_enhanced.append(se)

            if len(clean_list) < 12:
                clean_list.append(clean[i].cpu())
                noisy_list.append(noisy[i].cpu())
                masked_list.append(masked[i].cpu())
                enhanced_list.append(enhanced[i].cpu())
                label_list.append(labels[i].item())

        total_seen += clean.shape[0]
        if total_seen >= EVAL_IMAGES:
            break

psnr_noisy = np.array(psnr_noisy)
psnr_enhanced = np.array(psnr_enhanced)
ssim_noisy = np.array(ssim_noisy)
ssim_enhanced = np.array(ssim_enhanced)

psnr_gain = psnr_enhanced - psnr_noisy
ssim_gain = ssim_enhanced - ssim_noisy

print("=" * 70)
print("FINAL QUANTITATIVE RESULTS")
print("=" * 70)
print(f"Evaluation images: {len(psnr_noisy)}")
print(f"Average noisy PSNR     : {psnr_noisy.mean():.2f} dB")
print(f"Average enhanced PSNR  : {psnr_enhanced.mean():.2f} dB")
print(f"Average PSNR gain      : {psnr_gain.mean():+.2f} dB")
print(f"Average noisy SSIM     : {ssim_noisy.mean():.4f}")
print(f"Average enhanced SSIM  : {ssim_enhanced.mean():.4f}")
print(f"Average SSIM gain      : {ssim_gain.mean():+.4f}")
print("=" * 70)

metrics_table = np.column_stack([
    psnr_noisy,
    psnr_enhanced,
    psnr_gain,
    ssim_noisy,
    ssim_enhanced,
    ssim_gain
])

np.savetxt(
    f"{OUTPUT_DIR}/quantitative_metrics.csv",
    metrics_table,
    delimiter=",",
    header="PSNR_Noisy,PSNR_Enhanced,PSNR_Gain,SSIM_Noisy,SSIM_Enhanced,SSIM_Gain",
    comments=""
)


# ============================================================
# FIGURE 4: QUALITATIVE RESULTS
# ============================================================

def save_qualitative_results(num_samples=6):
    fig, axes = plt.subplots(num_samples, 4, figsize=(11, 2.6 * num_samples))

    col_titles = [
        "Original Clean",
        "Noisy Input",
        "Masked Input",
        "Enhanced Output"
    ]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for i in range(num_samples):
        row_images = [
            clean_list[i],
            noisy_list[i],
            masked_list[i],
            enhanced_list[i]
        ]

        clean_ref = clean_list[i]
        pn, sn = compute_single_metrics(clean_ref, noisy_list[i])
        pe, se = compute_single_metrics(clean_ref, enhanced_list[i])

        row_labels = [
            f"Class: {class_names[label_list[i]]}",
            f"PSNR: {pn:.2f} dB\nSSIM: {sn:.3f}",
            "75% patch masking",
            f"PSNR: {pe:.2f} dB\nSSIM: {se:.3f}"
        ]

        for j in range(4):
            axes[i, j].imshow(np.clip(tensor_to_numpy_image(row_images[j]), 0, 1))
            axes[i, j].set_xlabel(row_labels[j], fontsize=8)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    fig.suptitle(
        "Figure 4. Qualitative Image Enhancement Results",
        fontsize=15, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_qualitative_enhancement_results.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


save_qualitative_results()


# ============================================================
# FIGURE 5: PSNR AND SSIM COMPARISON
# ============================================================

def save_metric_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    metric_names = ["Noisy Input", "Enhanced Output"]

    psnr_means = [psnr_noisy.mean(), psnr_enhanced.mean()]
    psnr_stds = [psnr_noisy.std(), psnr_enhanced.std()]

    ssim_means = [ssim_noisy.mean(), ssim_enhanced.mean()]
    ssim_stds = [ssim_noisy.std(), ssim_enhanced.std()]

    axes[0].bar(metric_names, psnr_means, yerr=psnr_stds, capsize=6)
    axes[0].set_title("Average PSNR Comparison", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True, axis="y", alpha=0.3)

    for idx, value in enumerate(psnr_means):
        axes[0].text(idx, value + 0.15, f"{value:.2f}", ha="center", fontsize=10, fontweight="bold")

    axes[1].bar(metric_names, ssim_means, yerr=ssim_stds, capsize=6)
    axes[1].set_title("Average SSIM Comparison", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, axis="y", alpha=0.3)

    for idx, value in enumerate(ssim_means):
        axes[1].text(idx, value + 0.015, f"{value:.4f}", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle(
        "Figure 5. Quantitative Comparison Between Noisy and Enhanced Images",
        fontsize=15, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_psnr_ssim_comparison.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


save_metric_comparison()


# ============================================================
# FIGURE 6: PSNR AND SSIM GAIN DISTRIBUTION
# ============================================================

def save_gain_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].hist(psnr_gain, bins=25, alpha=0.85, edgecolor="black")
    axes[0].axvline(0, linestyle="--", linewidth=1.5)
    axes[0].axvline(psnr_gain.mean(), linestyle="-", linewidth=2.0)
    axes[0].set_title("PSNR Gain Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("PSNR Gain (dB)")
    axes[0].set_ylabel("Number of Images")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(ssim_gain, bins=25, alpha=0.85, edgecolor="black")
    axes[1].axvline(0, linestyle="--", linewidth=1.5)
    axes[1].axvline(ssim_gain.mean(), linestyle="-", linewidth=2.0)
    axes[1].set_title("SSIM Gain Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("SSIM Gain")
    axes[1].set_ylabel("Number of Images")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "Figure 6. Distribution of Enhancement Gains Across Test Images",
        fontsize=15, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig6_gain_distribution.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


save_gain_distribution()


# ============================================================
# FIGURE 7: NOISE LEVEL GENERALIZATION
# ============================================================

def evaluate_noise_level(std, max_images=150):
    local_psnr_noisy = []
    local_psnr_enhanced = []
    local_ssim_noisy = []
    local_ssim_enhanced = []

    model.eval()

    with torch.no_grad():
        seen = 0

        for imgs, _ in test_loader:
            clean = imgs.to(device)
            noisy = add_gaussian_noise(clean, std)
            masked, _, _ = mask_patches(noisy)
            enhanced = model(masked)

            for i in range(clean.shape[0]):
                pn, sn = compute_single_metrics(clean[i], noisy[i])
                pe, se = compute_single_metrics(clean[i], enhanced[i])

                local_psnr_noisy.append(pn)
                local_psnr_enhanced.append(pe)
                local_ssim_noisy.append(sn)
                local_ssim_enhanced.append(se)

            seen += clean.shape[0]
            if seen >= max_images:
                break

    return {
        "psnr_noisy": np.mean(local_psnr_noisy),
        "psnr_enhanced": np.mean(local_psnr_enhanced),
        "ssim_noisy": np.mean(local_ssim_noisy),
        "ssim_enhanced": np.mean(local_ssim_enhanced),
    }


def save_noise_generalization():
    noise_levels = [0.05, 0.10, 0.15, 0.20, 0.30]
    results = [evaluate_noise_level(std) for std in noise_levels]

    noisy_psnr_values = [r["psnr_noisy"] for r in results]
    enhanced_psnr_values = [r["psnr_enhanced"] for r in results]

    noisy_ssim_values = [r["ssim_noisy"] for r in results]
    enhanced_ssim_values = [r["ssim_enhanced"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(noise_levels, noisy_psnr_values, marker="o", linewidth=2.0, label="Noisy Input")
    axes[0].plot(noise_levels, enhanced_psnr_values, marker="o", linewidth=2.0, label="Enhanced Output")
    axes[0].set_title("PSNR Across Noise Levels", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Gaussian Noise Standard Deviation")
    axes[0].set_ylabel("Average PSNR (dB)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(noise_levels, noisy_ssim_values, marker="o", linewidth=2.0, label="Noisy Input")
    axes[1].plot(noise_levels, enhanced_ssim_values, marker="o", linewidth=2.0, label="Enhanced Output")
    axes[1].set_title("SSIM Across Noise Levels", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Gaussian Noise Standard Deviation")
    axes[1].set_ylabel("Average SSIM")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        "Figure 7. Generalization of Learned Representations Across Noise Levels",
        fontsize=15, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig7_noise_level_generalization.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


save_noise_generalization()


# ============================================================
# FIGURE 8: REPRESENTATION VISUALIZATION USING PCA
# ============================================================

def pca_2d(features):
    """
    Simple PCA implementation using NumPy SVD.
    This avoids requiring scikit-learn.
    """
    X = features - features.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


def save_representation_pca(max_images=800):
    model.eval()

    feature_bank = []
    label_bank = []

    with torch.no_grad():
        seen = 0

        for imgs, labels in test_loader:
            clean = imgs.to(device)
            noisy = add_gaussian_noise(clean, NOISE_STD)
            masked, _, _ = mask_patches(noisy)

            features = model.encode(masked)

            feature_bank.append(features.cpu().numpy())
            label_bank.append(labels.numpy())

            seen += clean.shape[0]
            if seen >= max_images:
                break

    features = np.concatenate(feature_bank, axis=0)
    labels = np.concatenate(label_bank, axis=0)

    coords = pca_2d(features)

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab10",
        s=16,
        alpha=0.75
    )

    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.ax.set_yticklabels(class_names)

    ax.set_title(
        "PCA Projection of Learned Encoder Representations",
        fontsize=12,
        fontweight="bold"
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Figure 8. Visualization of Self-Supervised Learned Representations",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig8_representation_pca.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


save_representation_pca()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ALL OUTPUTS SAVED SUCCESSFULLY")
print("=" * 70)
print(f"Output folder: {OUTPUT_DIR}")
print("")
print("Generated report figures:")
print("1. fig1_mae_workflow_architecture.png")
print("2. fig2_patch_masking_demo.png")
print("3. fig3_training_loss_curves.png")
print("4. fig4_qualitative_enhancement_results.png")
print("5. fig5_psnr_ssim_comparison.png")
print("6. fig6_gain_distribution.png")
print("7. fig7_noise_level_generalization.png")
print("8. fig8_representation_pca.png")
print("")
print("Additional files:")
print("mae_enhancer_model.pth")
print("quantitative_metrics.csv")
print("=" * 70)

print("\nRecommended result statement for report:")
print(
    f"The proposed MAE-based self-supervised enhancement model achieved "
    f"an average PSNR change of {psnr_gain.mean():+.2f} dB and an average "
    f"SSIM change of {ssim_gain.mean():+.4f} on {len(psnr_noisy)} CIFAR-10 "
    f"test images corrupted with Gaussian noise at sigma = {NOISE_STD}. "
    f"The results show that masked reconstruction can learn useful visual "
    f"representations for image enhancement, although fine details may still "
    f"be smoothed due to the low-resolution dataset and limited model capacity."
)
