import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from bihugan.config import load_config
from bihugan.data.volume_paired_dataset import PairedVolumeDataset, worker_init_fn
from bihugan.models.generator import BiHUGenerator
from bihugan.models.discriminator import PatchDiscriminator
from bihugan.losses.gan_lsgan import lsgan_d_loss, lsgan_g_loss
from bihugan.losses.cycle import cycle_l1_loss
from bihugan.losses.gradients import SobelGrad2D, gradient_consistency_loss
from bihugan.losses.hu_loss import HUConsistencyLoss
from bihugan.utils.metrics import (
    compute_ssim_psnr_masked,
    compute_nmae_masked,
    compute_hu_mad,
)


def set_seed(seed: int):
    """Set random seed for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cosine_warmup_lr(epoch, total_epochs, base_lr, final_lr=1e-6, warmup_epochs=5):
    """
    Cosine annealing with linear warmup.
    Returns learning rate for a given epoch.
    """
    if epoch < warmup_epochs:
        # Linear warmup phase
        return base_lr * (epoch + 1) / warmup_epochs
    # Cosine decay phase
    t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return final_lr + 0.5 * (base_lr - final_lr) * (1.0 + math.cos(math.pi * t))


def set_requires_grad(module, flag):
    """Enable or disable gradient computation for a module."""
    for p in module.parameters():
        p.requires_grad_(flag)


@torch.no_grad()
def validate(val_loader, device, G_A2B, G_B2A, hu_min, hu_max):
    """Run validation loop and compute metrics."""
    G_A2B.eval()
    G_B2A.eval()

    # Containers for metrics
    metrics = {
        "ssim_ab": [], "psnr_ab": [], "nmae_ab": [],
        "ssim_ba": [], "psnr_ba": [], "nmae_ba": [],
        "hu_mads_ba": [], "hu_mads_ab": []
    }

    for batch in val_loader:
        A = batch["A"].to(device)
        B = batch["B"].to(device)
        mask_A = batch["mask_A"].to(device)
        mask_B = batch["mask_B"].to(device)

        # Generate synthetic images
        sB = G_A2B(A)
        sA = G_B2A(B)

        # A -> B direction metrics
        ssim_ab, psnr_ab = compute_ssim_psnr_masked(B, sB, mask_B, data_range=2.0)
        nmae_ab = compute_nmae_masked(sB, B, mask_B)

        # B -> A direction metrics
        ssim_ba, psnr_ba = compute_ssim_psnr_masked(A, sA, mask_A, data_range=2.0)
        nmae_ba = compute_nmae_masked(sA, A, mask_A)

        # HU-based Mean Absolute Deviation
        hu_mad_ba = compute_hu_mad(A, sA, mask_body=mask_A, hu_min=hu_min, hu_max=hu_max)
        hu_mad_ab = compute_hu_mad(B, sB, mask_body=mask_B, hu_min=hu_min, hu_max=hu_max)

        # Store per-sample metrics
        metrics["ssim_ab"].append(ssim_ab)
        metrics["psnr_ab"].append(psnr_ab)
        metrics["nmae_ab"].append(nmae_ab)
        metrics["ssim_ba"].append(ssim_ba)
        metrics["psnr_ba"].append(psnr_ba)
        metrics["nmae_ba"].append(nmae_ba)
        metrics["hu_mads_ba"].append(hu_mad_ba)
        metrics["hu_mads_ab"].append(hu_mad_ab)

    # Return averaged metrics
    return (
        np.mean(metrics["ssim_ab"]), np.mean(metrics["psnr_ab"]), np.mean(metrics["nmae_ab"]),
        np.mean(metrics["ssim_ba"]), np.mean(metrics["psnr_ba"]), np.mean(metrics["nmae_ba"]),
        np.mean(metrics["hu_mads_ba"]), np.mean(metrics["hu_mads_ab"]),
    )


def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="BiHU-GAN Training")
    parser.add_argument("--config", type=str, default="configs/bihugan.yaml",
                        help="Path to configuration YAML file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Set device
    device = torch.device("cuda" if (cfg.get("cuda", True) and torch.cuda.is_available()) else "cpu")
    set_seed(int(cfg["seed"]))

    # Resolution consistency check
    internal_res = int(cfg["internal_resolution"])
    model_res = int(cfg["model_input_resolution"])
    if model_res != internal_res:
        raise ValueError(f"Training resolution must match internal_resolution. Got {model_res} vs {internal_res}")

    # ---------------- Datasets ----------------
    train_dataset = PairedVolumeDataset(
        list_file=cfg["train_list"],
        pair_from=cfg["pair_from"],
        pair_to=cfg["pair_to"],
        hu_min=cfg["hu_min"],
        hu_max=cfg["hu_max"],
        body_hu_thresh=cfg["body_hu_thresh"],
        bone_hu_thresh=cfg["bone_hu_thresh"],
        model_input_resolution=model_res,
        upsample_to_model=True,
        enable_augmentation=True,
        rng_seed=int(cfg["seed"]),
    )
    if len(train_dataset) == 0:
        raise RuntimeError(f"Training dataset is empty! Check {cfg['train_list']} and DICOM paths.")

    val_dataset = PairedVolumeDataset(
        list_file=cfg["val_list"],
        pair_from=cfg["pair_from"],
        pair_to=cfg["pair_to"],
        hu_min=cfg["hu_min"],
        hu_max=cfg["hu_max"],
        body_hu_thresh=cfg["body_hu_thresh"],
        bone_hu_thresh=cfg["bone_hu_thresh"],
        model_input_resolution=model_res,
        upsample_to_model=True,
        enable_augmentation=False,   # No augmentation during validation
        rng_seed=int(cfg["seed"]),
    )

    # ---------------- Data Loaders ----------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batchSize"],
        shuffle=True,
        num_workers=cfg["num_workers_train"],
        pin_memory=True,
        worker_init_fn=worker_init_fn if cfg["num_workers_train"] > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["num_workers_val"],
        pin_memory=True,
    )

    # ---------------- Models ----------------
    mc_p = float(cfg["mc_dropout_p_train"])
    G_A2B = BiHUGenerator(in_ch=1, out_ch=1, base=64, n_res=9, mc_dropout_p=mc_p).to(device)
    G_B2A = BiHUGenerator(in_ch=1, out_ch=1, base=64, n_res=9, mc_dropout_p=mc_p).to(device)
    D_A = PatchDiscriminator(in_ch=1, base=64).to(device)
    D_B = PatchDiscriminator(in_ch=1, base=64).to(device)

    # ---------------- Optimizers ----------------
    opt_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()),
                       lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_D = optim.Adam(list(D_A.parameters()) + list(D_B.parameters()),
                       lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    # ---------------- Loss functions ----------------
    lam = float(cfg["global_scaling_factor"])            # weight for GAN+cycle+gradient losses
    w_hu = float(cfg["hu_loss_weight"])                  # weight for HU consistency loss
    hu_loss_fn = HUConsistencyLoss(
        hu_min=cfg["hu_min"], hu_max=cfg["hu_max"],
        body_hu_thresh=cfg["body_hu_thresh"], bone_hu_thresh=cfg["bone_hu_thresh"],
        w_soft=1.0, w_bone=1.5,
    )
    grad_op = SobelGrad2D().to(device)                   # Sobel operator for gradient consistency

    # Automatic Mixed Precision
    use_amp = cfg.get("amp", False) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Training configuration
    epochs = int(cfg["epochs"])
    grad_accum_steps = int(cfg["grad_accum_steps"])
    early_stopping_enabled = cfg.get("early_stopping_enabled", True)
    patience = int(cfg.get("early_stopping_patience", 20))
    best_metric = float("inf")
    epochs_no_improve = 0
    best_epoch = 0

    # Learning rate scheduler function
    def get_lr(epoch):
        return cosine_warmup_lr(epoch, epochs, cfg["lr"], final_lr=1e-6, warmup_epochs=5)

    val_frequency = 5            # Validate every N epochs
    save_latest_every = 10       # Save a checkpoint every N epochs

    # ===================== TRAINING LOOP =====================
    for epoch in range(1, epochs + 1):
        G_A2B.train(); G_B2A.train()
        D_A.train(); D_B.train()

        # Set learning rate for this epoch
        lr = get_lr(epoch - 1)
        for pg in opt_G.param_groups: pg["lr"] = lr
        for pg in opt_D.param_groups: pg["lr"] = lr

        epoch_G_loss = 0.0
        epoch_D_loss = 0.0
        num_batches = len(train_loader)

        # Zero out gradients for accumulators
        opt_D.zero_grad(set_to_none=True)
        opt_G.zero_grad(set_to_none=True)
        accum_step = 0

        for step, batch in enumerate(train_loader, 1):
            A = batch["A"].to(device)
            B = batch["B"].to(device)
            mask_A = batch["mask_A"].to(device)

            # ------------------------------------------------------------------
            # Discriminator update
            # ------------------------------------------------------------------
            set_requires_grad(D_A, True); set_requires_grad(D_B, True)
            set_requires_grad(G_A2B, False); set_requires_grad(G_B2A, False)

            with torch.no_grad():
                sB_det = G_A2B(A)          # fake B
                sA_det = G_B2A(B)          # fake A

            with autocast(enabled=use_amp):
                loss_D = 0.5 * (lsgan_d_loss(D_A(A), D_A(sA_det)) +
                                lsgan_d_loss(D_B(B), D_B(sB_det)))
            scaler.scale(loss_D / grad_accum_steps).backward()

            # ------------------------------------------------------------------
            # Generator update
            # ------------------------------------------------------------------
            set_requires_grad(D_A, False); set_requires_grad(D_B, False)
            set_requires_grad(G_A2B, True); set_requires_grad(G_B2A, True)

            with autocast(enabled=use_amp):
                sB = G_A2B(A)
                sA = G_B2A(B)
                rec_A = G_B2A(sB)           # cycle consistency: A -> fake B -> rec A
                rec_B = G_A2B(sA)           # B -> fake A -> rec B

                loss_G = (
                    lam * (
                        # Adversarial loss
                        10.0 * (lsgan_g_loss(D_A(sA)) + lsgan_g_loss(D_B(sB))) +
                        # Identity-like cycle loss on synthetic images
                        5.0  * (cycle_l1_loss(sB, B) + cycle_l1_loss(sA, A)) +
                        # Full cycle consistency loss
                        1.0  * (cycle_l1_loss(rec_A, A) + cycle_l1_loss(rec_B, B)) +
                        # Gradient consistency loss
                        10.0 * (gradient_consistency_loss(sB, B, grad_op) +
                                gradient_consistency_loss(sA, A, grad_op))
                    )
                    # HU consistency loss
                    + w_hu * hu_loss_fn(A, sA, mask_body=mask_A)
                )
            scaler.scale(loss_G / grad_accum_steps).backward()

            accum_step += 1
            # Perform optimizer step when accumulation steps are reached or at the end of epoch
            if accum_step % grad_accum_steps == 0 or step == num_batches:
                scaler.step(opt_D)
                scaler.step(opt_G)
                scaler.update()
                opt_D.zero_grad(set_to_none=True)
                opt_G.zero_grad(set_to_none=True)
                accum_step = 0

            # Accumulate loss for logging
            epoch_D_loss += loss_D.item()
            epoch_G_loss += loss_G.item()

        avg_G = epoch_G_loss / num_batches
        avg_D = epoch_D_loss / num_batches

        # Validation and logging
        if epoch % val_frequency == 0:
            ssim_ab, psnr_ab, nmae_ab, ssim_ba, psnr_ba, nmae_ba, hu_mad_ba, hu_mad_ab = validate(
                val_loader, device, G_A2B, G_B2A, cfg["hu_min"], cfg["hu_max"]
            )
            print(
                f"Epoch {epoch:3d}/{epochs} | LR {lr:.2e} | "
                f"G Loss: {avg_G:.4f} | D Loss: {avg_D:.4f}\n"
                f"A→B SSIM(m):{ssim_ab:.4f} PSNR(m):{psnr_ab:.2f} NMAE(m):{nmae_ab:.4f}\n"
                f"B→A SSIM(m):{ssim_ba:.4f} PSNR(m):{psnr_ba:.2f} NMAE(m):{nmae_ba:.4f}\n"
                f"HU-MAD(m) B→A:{hu_mad_ba:.2f} A→B:{hu_mad_ab:.2f} HU"
            )

            # Early stopping logic based on combined metric
            if early_stopping_enabled:
                if np.isnan(ssim_ab) or np.isnan(hu_mad_ba) or np.isinf(hu_mad_ba):
                    epochs_no_improve += 1
                else:
                    # Normalize the components and combine
                    ssim_error = 1.0 - ssim_ab
                    hu_norm_ba = hu_mad_ba / (hu_mad_ba + 100.0)
                    curr = 0.5 * ssim_error + 0.5 * hu_norm_ba

                    if curr < best_metric:
                        best_metric = curr
                        epochs_no_improve = 0
                        best_epoch = epoch
                        # Save best models
                        os.makedirs(cfg["checkpoints_dir"], exist_ok=True)
                        torch.save(G_A2B.state_dict(), os.path.join(cfg["checkpoints_dir"], "G_A2B_best.pth"))
                        torch.save(G_B2A.state_dict(), os.path.join(cfg["checkpoints_dir"], "G_B2A_best.pth"))
                        torch.save(D_A.state_dict(), os.path.join(cfg["checkpoints_dir"], "D_A_best.pth"))
                        torch.save(D_B.state_dict(), os.path.join(cfg["checkpoints_dir"], "D_B_best.pth"))
                    else:
                        epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch} epochs. Best epoch: {best_epoch}")
                    break
        else:
            print(f"Epoch {epoch:3d}/{epochs} | LR {lr:.2e} | G Loss: {avg_G:.4f} | D Loss: {avg_D:.4f}")

        # Periodic saving of latest models
        if epoch % save_latest_every == 0:
            ckpt_dir = cfg["checkpoints_dir"]
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(G_A2B.state_dict(), os.path.join(ckpt_dir, "G_A2B_latest.pth"))
            torch.save(G_B2A.state_dict(), os.path.join(ckpt_dir, "G_B2A_latest.pth"))
            torch.save(D_A.state_dict(), os.path.join(ckpt_dir, "D_A_latest.pth"))
            torch.save(D_B.state_dict(), os.path.join(ckpt_dir, "D_B_latest.pth"))

    print("Training finished.")


if __name__ == "__main__":
    main()