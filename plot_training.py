import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib and numpy are required for live plotting. Install them with: pip install matplotlib numpy")
    sys.exit(1)

from sbox import train as train_wgan
from dcgan import train_dcgan


def main(model="wgan", resume=True, dataset_path=None, epochs_num=50, batch_size=64, learning_rate=2e-4, beta1=0.5):
    epochs = []
    g_losses = []
    d_losses = []
    du_values = []
    nl_values = []
    bij_values = []

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    ax_loss, ax_diff, ax_other = axes
    ax_loss.set_title("GAN Training Metrics")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True)

    ax_diff.set_title("Differential Uniformity / Nonlinearity")
    ax_diff.set_ylabel("Value")
    ax_diff.grid(True)

    ax_other.set_title("Bijection Loss")
    ax_other.set_ylabel("Value")
    ax_other.set_xlabel("Epoch")
    ax_other.grid(True)

    g_line, = ax_loss.plot([], [], label="Generator Loss", color="tab:blue")
    d_line, = ax_loss.plot([], [], label="Discriminator Loss", color="tab:orange")
    du_line, = ax_diff.plot([], [], label="Differential Uniformity", color="tab:red")
    nl_line, = ax_diff.plot([], [], label="Nonlinearity", color="tab:green")
    bij_line, = ax_other.plot([], [], label="Bijection Loss", color="tab:purple")

    ax_loss.legend(loc="upper right")
    ax_diff.legend(loc="upper right")
    ax_other.legend(loc="upper right")

    def progress_callback(epoch, g_loss, d_loss, du_value, nl_value, bij_value):
        epochs.append(epoch + 1)
        g_losses.append(g_loss)
        d_losses.append(d_loss)
        du_values.append(du_value if du_value is not None else np.nan)
        nl_values.append(nl_value if nl_value is not None else np.nan)
        bij_values.append(bij_value if bij_value is not None else np.nan)

        g_line.set_data(epochs, g_losses)
        d_line.set_data(epochs, d_losses)
        du_line.set_data(epochs, du_values)
        nl_line.set_data(epochs, nl_values)
        bij_line.set_data(epochs, bij_values)

        for ax in axes:
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

    print("Starting training with live matplotlib plotting...")
    if model == "dcgan":
        train_dcgan(
            z_dim=128,
            epochs=epochs_num,
            batch_size=batch_size,
            lr=learning_rate,
            beta1=beta1,
            dataset_path=dataset_path,
            progress_callback=progress_callback,
        )
    else:
        train_wgan(resume=resume, progress_callback=progress_callback)

    plt.ioff()
    print("Training complete. Close the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
