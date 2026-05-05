import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from bihugan.config import load_config
from bihugan.data.volume_paired_dataset import PairedVolumeDataset
from bihugan.models.generator import BiHUGenerator
from bihugan.utils.export_dicom import save_dicom_series_slice
from bihugan.utils.metrics import compute_ssim_psnr_masked, compute_hu_mad


@torch.no_grad()
def run_inference(config_path, direction="B2A", export=True):
    """
    Main inference routine.

    Args:
        config_path (str): Path to the YAML configuration file.
        direction (str): Translation direction, either 'A2B' or 'B2A'.
        export (bool): Whether to save results as DICOM series.
    """
    # Load configuration
    cfg = load_config(config_path)
    device = torch.device("cuda" if (cfg.get("cuda", True) and torch.cuda.is_available()) else "cpu")

    # Build dataset from test list (no augmentation)
    dataset = PairedVolumeDataset(
        list_file=cfg["test_list"],
        pair_from=cfg["pair_from"],
        pair_to=cfg["pair_to"],
        hu_min=cfg["hu_min"],
        hu_max=cfg["hu_max"],
        body_hu_thresh=cfg["body_hu_thresh"],
        bone_hu_thresh=cfg["bone_hu_thresh"],
        model_input_resolution=int(cfg["model_input_resolution"]),
        upsample_to_model=True,
        enable_augmentation=False,
        rng_seed=int(cfg["seed"]),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create both generators with inference dropout rate
    mc_p_infer = float(cfg.get("mc_dropout_p_infer", 0.0))
    G_A2B = BiHUGenerator(in_ch=1, out_ch=1, base=64, n_res=9, mc_dropout_p=mc_p_infer).to(device)
    G_B2A = BiHUGenerator(in_ch=1, out_ch=1, base=64, n_res=9, mc_dropout_p=mc_p_infer).to(device)

    # Load best checkpoint weights
    checkpoint_dir = cfg["checkpoints_dir"]
    best_A2B = os.path.join(checkpoint_dir, "G_A2B_best.pth")
    best_B2A = os.path.join(checkpoint_dir, "G_B2A_best.pth")
    if not os.path.exists(best_A2B) or not os.path.exists(best_B2A):
        raise FileNotFoundError(f"Checkpoint not found: {best_A2B} or {best_B2A}")

    G_A2B.load_state_dict(torch.load(best_A2B, map_location=device))
    G_B2A.load_state_dict(torch.load(best_B2A, map_location=device))

    # Set dropout behaviour: train() keeps MC dropout active; eval() disables it
    if mc_p_infer > 0:
        G_A2B.train()
        G_B2A.train()
    else:
        G_A2B.eval()
        G_B2A.eval()

    # Select generator and tag according to direction
    active_G = G_A2B if direction == "A2B" else G_B2A
    export_tag = cfg["export_series_tag_A2B"] if direction == "A2B" else cfg["export_series_tag_B2A"]

    patient_uids = {}           # Per-patient DICOM UID cache
    metrics_ssim = []           # SSIM values for each slice
    metrics_hu_mad = []         # HU-MAD values for each slice

    # Inference loop over all test slices
    for batch in loader:
        A = batch["A"].to(device)
        B = batch["B"].to(device)
        mask_A = batch["mask_A"].to(device)
        mask_B = batch["mask_B"].to(device)

        # Determine source, reference and evaluation mask based on direction
        template_path = batch["B_path"][0] if direction == "A2B" else batch["A_path"][0]
        inp, ref = (A, B) if direction == "A2B" else (B, A)
        eval_mask = mask_B if direction == "A2B" else mask_A

        # Generate synthetic image
        s_out = active_G(inp)

        # Compute quality metrics (SSIM, HU-MAD)
        ssim_val, _ = compute_ssim_psnr_masked(ref, s_out, eval_mask, data_range=2.0)
        metrics_ssim.append(ssim_val)

        hu_mad_val = compute_hu_mad(ref, s_out, mask_body=eval_mask,
                                    hu_min=cfg["hu_min"], hu_max=cfg["hu_max"])
        metrics_hu_mad.append(hu_mad_val)

        # Optionally export DICOM slices
        if export:
            # Extract series_id and slice index for naming
            series_id = batch.get("series_id", [0])[0]
            if torch.is_tensor(series_id):
                series_id = series_id.item()
            slice_idx = batch.get("slice_idx", [0])[0]
            if torch.is_tensor(slice_idx):
                slice_idx = slice_idx.item()

            patient_id = f"case_{int(series_id):03d}"
            save_dicom_series_slice(
                image_norm=s_out[0, 0].cpu().numpy(),
                template_dcm_path=template_path,
                patient_id=patient_id,
                series_description_tag=export_tag,
                instance_num=int(slice_idx) + 1,
                output_root=cfg["output_root"],
                hu_min=cfg["hu_min"],
                hu_max=cfg["hu_max"],
                patient_uids=patient_uids,
            )

    # Summary statistics
    print(f"Inference completed. Direction: {direction}")
    if metrics_ssim:
        print(f"Mean SSIM (masked): {np.mean(metrics_ssim):.4f}")
    if metrics_hu_mad:
        print(f"Mean HU-MAD (masked): {np.mean(metrics_hu_mad):.2f} HU")


if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(description="BiHU-GAN Inference/Test")
    parser.add_argument("--config", type=str, default="configs/bihugan.yaml",
                        help="Path to YAML configuration")
    parser.add_argument("--direction", type=str, default=None,
                        help="Translation direction: A2B or B2A (overrides config)")
    parser.add_argument("--no_export", action="store_true",
                        help="Disable DICOM export")
    args = parser.parse_args()

    # Resolve direction from CLI or config
    cfg = load_config(args.config)
    direction = args.direction if args.direction is not None else cfg.get("inference_direction", "B2A")

    # Run inference
    run_inference(args.config, direction=direction, export=not args.no_export)