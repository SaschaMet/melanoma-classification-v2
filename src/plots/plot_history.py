import matplotlib.pyplot as plt


def plot_simple_history(history, metric, include_validation=False):
    # summarize history for accuracy
    plt.plot(history.history[metric])
    if include_validation:
        plt.plot(history.history['val_auc'])
    plt.title('model auc')
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    if include_validation:
        plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def compare_historys(original_history, new_history, initial_epochs, metric="accuracy", include_validation=False):
    """
    Compares two model history objects.
    """
    # Get original history measurements
    acc = original_history.history[metric]
    loss = original_history.history["loss"]

    if include_validation:
        val_acc = original_history.history["val_" + metric]
        val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history[metric]
    total_loss = loss + new_history.history["loss"]

    if include_validation:
        total_val_acc = val_acc + new_history.history["val_" + metric]
        total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_loss, label='Training Loss')
    if include_validation:
        plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(total_acc, label='Training ' + metric.title())
    if include_validation:
        plt.plot(total_val_acc, label='Validation ' + metric.title())
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('' + metric.title())

    plt.xlabel('epoch')
    plt.show()
