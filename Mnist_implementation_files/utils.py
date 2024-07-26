# utils.py

import torch
import matplotlib.pyplot as plt
import einops
import imageio 
def save_plot_images(images, title, filename):
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    fig.suptitle(title, fontsize=16)
    for i, (label, image) in enumerate(images.items()):
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f'Class {label}')
        axes[i].axis('off')
    plt.savefig(filename)
    plt.close(fig)


def show_images2(images, title=""):
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                ax.imshow(images[idx][0], cmap="gray")
                ax.axis('off')  # Turn off axis
                idx += 1

    fig.suptitle(title, fontsize=30)
    plt.show()



def show_images(images, title=""):
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.show()

def get_images_by_class(loader):
    class_images = {}
    for images, labels in loader:
        for i in range(len(labels)):
            label = labels[i].item()
            if label not in class_images:
                class_images[label] = images[i]
            if len(class_images) == 10:
                return class_images
    return class_images

def plot_images(images, title):
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    fig.suptitle(title, fontsize=16)
    for i, (label, image) in enumerate(images.items()):
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f'Class {label}')
        axes[i].axis('off')
    plt.show()


