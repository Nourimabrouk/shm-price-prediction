import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    out = Path("test_output")
    out.mkdir(exist_ok=True)
    fig_path = out / "smoke_plot.png"
    plt.figure()
    plt.plot([0, 1, 2], [0, 1, 4], label="smoke")
    plt.title("Visualization Smoke Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()

