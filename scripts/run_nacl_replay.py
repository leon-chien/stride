from __future__ import annotations

from stride.replay.nacl_replay import run_nacl_replay


def main() -> None:
    results = run_nacl_replay(
        dataset_path="outputs/nacl/nacl_dataset.npz",
        checkpoint_path="checkpoints/nacl_gru.pt",
        batch_size=512,
        num_bins=4,
        top_k=0.10,
        seed=42,
    )

    metrics = results["metrics"]

    print("\nNaCl replay results:")
    print(f"Overall positive rate: {metrics['positive_rate']:.4f}")
    print(f"Random top 10% positive rate: {metrics['random_top10_positive_rate']:.4f}")
    print(f"Model top 10% positive rate: {metrics['top10_positive_rate']:.4f}")
    print(f"Top 10% enrichment: {metrics['top10_enrichment']:.2f}x")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")

    print("\nBin summary:")
    print(
        f"{'bin':>5} | {'n':>8} | {'frac':>8} | "
        f"{'pos_rate':>10} | {'mean_score':>10} | {'min_score':>10} | {'max_score':>10}"
    )
    print("-" * 86)

    for row in results["bin_summary"]:
        print(
            f"{int(row['bin_id']):>5} | "
            f"{int(row['n']):>8} | "
            f"{row['fraction']:>8.3f} | "
            f"{row['positive_rate']:>10.4f} | "
            f"{row['mean_score']:>10.4f} | "
            f"{row['min_score']:>10.4f} | "
            f"{row['max_score']:>10.4f}"
        )


if __name__ == "__main__":
    main()