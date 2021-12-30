CLASSES = {
    "0": 'unknown',
    "1": 'nevus',
    "2": 'melanoma',
    "3": 'seborrheic keratosis',
    "4": 'lentigo NOS',
    "5": 'lichenoid keratosis',
    "6": 'solar lentigo',
    "7": 'cafe-au-lait macule',
    "8": 'atypical melanocytic proliferation',
    "11": 'basal cell carcinoma',
    "12": 'actinic keratosis',
    "14": 'dermatofibroma',
    "15": 'vascular lesion',
    "16": 'squamous cell carcinoma',
}

CLASS_MAPPING = {
    "-1": "0",
    "9": '2',
    "10": '1',
    "13": '6',
    "17": '0',
}


def map_classes(label):
    label_string = str(label)
    if label_string in CLASS_MAPPING.keys():
        return int(CLASS_MAPPING.get(label_string))
    return label


for i in range(0, 18):
    label = map_classes(i)
    label_string = CLASSES.get(str(label))
    print(label, label_string)
    assert label in [int(x) for x in CLASSES.keys()]
