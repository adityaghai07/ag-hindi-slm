import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class TrainLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "train_log.json")
        self.plot_path = os.path.join(log_dir, "loss_curve.png")
        self.records: list[dict] = []

        # resume existing log if present
        if os.path.isfile(self.log_path):
            with open(self.log_path) as f:
                self.records = json.load(f)

    def log(self, step: int, loss: float, lr: float, grad_norm: float, tok_per_sec: float, vram_gb: float):
        self.records.append({
            "step": step,
            "loss": round(loss, 6),
            "lr": lr,
            "grad_norm": round(grad_norm, 4),
            "tok_per_sec": round(tok_per_sec, 1),
            "vram_gb": round(vram_gb, 2),
        })
        with open(self.log_path, "w") as f:
            json.dump(self.records, f, indent=2)

    def plot(self):
        if len(self.records) < 2:
            return

        steps = [r["step"] for r in self.records]
        losses = [r["loss"] for r in self.records]
        lrs = [r["lr"] for r in self.records]
        tok_s = [r["tok_per_sec"] for r in self.records]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.patch.set_facecolor("#0f0f0f")

        def style_ax(ax, title, ylabel):
            ax.set_facecolor("#1a1a1a")
            ax.set_title(title, color="#e0e0e0", fontsize=12, pad=8)
            ax.set_ylabel(ylabel, color="#a0a0a0", fontsize=10)
            ax.set_xlabel("step", color="#a0a0a0", fontsize=10)
            ax.tick_params(colors="#a0a0a0")
            ax.spines[:].set_color("#333333")
            ax.grid(True, color="#2a2a2a", linewidth=0.8, linestyle="--")
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_color("#a0a0a0")

        # --- loss ---
        ax = axes[0]
        ax.plot(steps, losses, color="#4fc3f7", linewidth=1.2, alpha=0.4, label="raw")
        if len(losses) >= 50:
            smooth = _smooth(losses, window=50)
            ax.plot(steps, smooth, color="#0288d1", linewidth=2.0, label="smoothed")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        style_ax(ax, "Training Loss", "loss")
        ax.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="#e0e0e0", fontsize=9)

        # --- lr ---
        ax = axes[1]
        ax.plot(steps, lrs, color="#81c784", linewidth=1.8)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))
        style_ax(ax, "Learning Rate", "lr")

        # --- throughput ---
        ax = axes[2]
        tok_k = [t / 1000 for t in tok_s]
        ax.plot(steps, tok_k, color="#ffb74d", linewidth=1.2, alpha=0.5)
        if len(tok_k) >= 20:
            ax.plot(steps, _smooth(tok_k, window=20), color="#f57c00", linewidth=2.0)
        style_ax(ax, "Throughput", "tok/s (K)")

        plt.tight_layout(pad=2.0)
        plt.savefig(self.plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()


def _smooth(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        out.append(sum(values[start:end]) / (end - start))
    return out
