"""
train.py
========
Command-line training script for Ptychographic reconstruction with Neural Operators.

Usage example
-------------
    python train.py \\
        --data_dir ./gdrive/MyDrive/PtychoNNData/ \\
        --model_type tfno \\
        --n_modes 24 24 \\
        --hidden_channels 64 \\
        --n_layers 4 \\
        --factorization tucker \\
        --rank 0.1 \\
        --batch_size 64 \\
        --lr 1e-3 \\
        --n_epochs 100 \\
        --eval_interval 5 \\
        --save_path best_model.pt

All arguments have sensible defaults so the script can be run without
any flags for a quick sanity check (provided the data directory is set).
"""

import argparse

import torch
from neuralop import LpLoss, H1Loss, Trainer
from neuralop.data.transforms.data_processors import IncrementalDataProcessor
from neuralop.training import AdamW

from data_setup import create_dataloaders
from models import build_model


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments for model training.

    Returns:
        Namespace object with all parsed hyperparameters and paths.
    """
    parser = argparse.ArgumentParser(
        description="Train a Neural Operator for Ptychographic reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data ---
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data_dir",
        type=str,
        default="./gdrive/MyDrive/PtychoNNData/",
        help="Directory containing the .npy data files.",
    )
    data_group.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size for training and validation.",
    )
    data_group.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )

    # --- Model architecture ---
    model_group = parser.add_argument_group("Model architecture")
    model_group.add_argument(
        "--model_type",
        type=str,
        default="tfno",
        choices=["fno", "tfno"],
        help="Neural Operator variant to train.",
    )
    model_group.add_argument(
        "--n_modes",
        type=int,
        nargs=2,
        default=[24, 24],
        metavar=("NX", "NY"),
        help="Number of Fourier modes to keep per spatial dimension.",
    )
    model_group.add_argument(
        "--in_channels",
        type=int,
        default=1,
        help="Number of input channels (diffraction patterns).",
    )
    model_group.add_argument(
        "--out_channels",
        type=int,
        default=2,
        help="Number of output channels (amplitude + phase).",
    )
    model_group.add_argument(
        "--hidden_channels",
        type=int,
        default=64,
        help="Number of hidden channels in FNO layers.",
    )
    model_group.add_argument(
        "--projection_channel_ratio",
        type=int,
        default=2,
        help="Ratio of projection MLP hidden size to hidden_channels.",
    )
    model_group.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="Number of FNO blocks.",
    )
    model_group.add_argument(
        "--positional_embedding",
        type=str,
        default="grid",
        choices=["grid", "none"],
        help="Positional embedding type. Use 'none' to disable.",
    )

    # --- TFNO-specific ---
    tfno_group = parser.add_argument_group("TFNO-specific (ignored for FNO)")
    tfno_group.add_argument(
        "--factorization",
        type=str,
        default="tucker",
        choices=["tucker", "cp", "tt"],
        help="Tensor factorization scheme for spectral weights.",
    )
    tfno_group.add_argument(
        "--rank",
        type=float,
        default=0.1,
        help="Compression rank. Float in (0,1] for relative rank.",
    )

    # --- Optimisation ---
    optim_group = parser.add_argument_group("Optimisation")
    optim_group.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for AdamW.",
    )
    optim_group.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularisation) for AdamW.",
    )
    optim_group.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Total number of training epochs.",
    )
    optim_group.add_argument(
        "--eval_interval",
        type=int,
        default=5,
        help="Evaluate on the validation set every N epochs.",
    )

    # --- Misc ---
    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument(
        "--save_path",
        type=str,
        default="best_model.pt",
        help="File path to save the best model checkpoint.",
    )
    misc_group.add_argument(
        "--wandb_log",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> dict:
    """Set up and run a full training run based on parsed arguments.

    Steps performed:
        1. Select compute device (CUDA if available, else CPU).
        2. Build train / val DataLoaders.
        3. Instantiate the Neural Operator model.
        4. Configure AdamW optimiser and cosine annealing scheduler.
        5. Define training loss (L2) and evaluation losses (L2 + H1).
        6. Run training via the neuraloperator Trainer.
        7. Save the final model checkpoint.

    Args:
        args: Parsed command-line arguments (from :func:`parse_args`).

    Returns:
        Dict with final training metrics returned by the Trainer.
    """
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 2. Data
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 3. Model
    positional_embedding = None if args.positional_embedding == "none" else args.positional_embedding
    model = build_model(
        model_type=args.model_type,
        n_modes=tuple(args.n_modes),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        projection_channel_ratio=args.projection_channel_ratio,
        n_layers=args.n_layers,
        factorization=args.factorization,
        rank=args.rank,
        positional_embedding=positional_embedding,
    )
    model = model.to(device)

    # 4. Optimiser & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs
    )

    # 5. Losses
    l2_loss = LpLoss(d=2, p=2)
    h1_loss = H1Loss(d=2)
    train_loss = l2_loss
    eval_losses = {"h1": h1_loss, "l2": l2_loss}

    print(f"\nTrain loss : {train_loss}")
    print(f"Eval  losses : {eval_losses}\n")

    # 6. Data processor (no normalisation; adjust if needed)
    data_processor = IncrementalDataProcessor(
        in_normalizer=None,
        out_normalizer=None,
        device=device,
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        n_epochs=args.n_epochs,
        device=device,
        data_processor=data_processor,
        wandb_log=args.wandb_log,
        eval_interval=args.eval_interval,
        use_distributed=False,
        verbose=True,
    )

    results = trainer.train(
        train_loader=train_loader,
        test_loaders={"val": val_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

    # 8. Save checkpoint
    torch.save(model.state_dict(), args.save_path)
    print(f"\nModel saved to: {args.save_path}")

    return results

if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("Ptychography Neural Operator — Training Configuration")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key:<30} {value}")
    print("=" * 60 + "\n")

    results = train(args)

    print("\nFinal training metrics:")
    for key, value in results.items():
        print(f"  {key}: {value}")

