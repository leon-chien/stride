from __future__ import annotations

from stride.training.train_nacl import train_nacl


def main() -> None:
    train_nacl(
        config_path="configs/nacl.yaml",
        dataset_path="outputs/nacl/nacl_dataset.npz",
        checkpoint_path="checkpoints/nacl_gru.pt",
    )


if __name__ == "__main__":
    main()