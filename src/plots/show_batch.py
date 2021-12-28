import matplotlib.pyplot as plt


def show_batch(image_batch, label_batch, show_title=True):
    plt.figure(figsize=(6, 6))
    for n in range(9):
        ax = plt.subplot(3, 3, n + 1)
        plt.imshow(image_batch[n])
        if show_title:
            title = "Benign" if label_batch[n] == 0 else "Malignant"
            plt.title(title)
        plt.axis("off")
