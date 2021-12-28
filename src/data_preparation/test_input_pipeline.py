import numpy as np
from plots.show_batch import show_batch


def test_input_pipeline(train_ds, val_ds, standardize=False, show_images=True):

    divide_by = 255 if standardize else 1

    image_batch, label_batch = next(iter(train_ds))

    print('image_batch.shape: ', image_batch.shape)
    print('label_batch.shape: ', label_batch.shape)

    image_batch = image_batch.numpy()
    label_batch = label_batch.numpy()

    assert label_batch[0].dtype == "int32"
    assert label_batch[0] == 0 or label_batch[0] == 1

    assert image_batch[0].dtype == "float32"

    print("max test image", image_batch[0].max())
    print("min test image", image_batch[0].min())

    if show_images:
        show_batch(np.clip(image_batch / divide_by, 0, 1), label_batch)

    image_batch, label_batch = next(iter(val_ds))

    print('image_batch.shape: ', image_batch.shape)
    print('label_batch.shape: ', label_batch.shape)

    image_batch = image_batch.numpy()
    label_batch = label_batch.numpy()

    assert label_batch[0].dtype == "int32"
    assert label_batch[0] == 0 or label_batch[0] == 1

    assert image_batch[0].dtype == "float32"
    print("max val image", image_batch[0].max())
    print("min val image", image_batch[0].min())

    if show_images:
        show_batch(image_batch / divide_by, label_batch)
