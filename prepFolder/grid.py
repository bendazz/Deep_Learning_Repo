import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

_images = []

def save_image_grid(batch, batch_size=10, path='sample.png'):
    for img in batch:
        _images.append(img.squeeze().numpy())

    n = len(_images)
    cols = min(n, batch_size)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < n:
                axes[i][j].imshow(_images[idx], cmap='gray')
            axes[i][j].axis('off')

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
