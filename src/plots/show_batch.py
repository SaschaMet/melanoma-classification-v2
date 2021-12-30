import matplotlib.pyplot as plt
from utils.class_to_string import class_to_string


def show_batch(image_batch, label_batch, show_title=True):
    plt.figure(figsize=(6, 6))
    for n in range(9):
        ax = plt.subplot(3, 3, n + 1)
        plt.imshow(image_batch[n])
        if show_title:
            title = class_to_string(label_batch[n])
            plt.title(title)
        plt.axis("off")
