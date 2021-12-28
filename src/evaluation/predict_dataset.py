def predict_on_dataset(model, dataset):
    print("start predicting ...")
    labels = []
    predictions = []

    for image_batch, label_batch in iter(dataset):
        labels.append(label_batch.numpy())
        batch_predictions = model.predict(image_batch)
        predictions.append(batch_predictions)

    # flatten the lists
    labels = [item for sublist in labels for item in sublist]
    predictions = [item[0] for sublist in predictions for item in sublist]
    return predictions, labels
