import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from pathlib import Path
from data_preprocessing import CHANNELS, preprocess_input

ROOT = Path(__file__).resolve().parents[1]
BIN_PATH = Path(ROOT, 'bin')
OUTPUT_PATH = Path(ROOT, 'output')
DATASET_PATH = Path(ROOT, 'dataset')
IMAGES_PATH = Path(DATASET_PATH, 'images')

mapping_dict = {
    0: "Inactivate",
    1: "Activity",
}


def show_predict_mapping(decode, mapping_dict):
    """
    decode: list 
    mapping_dict: dict

    Eg: [0 0 1 0 1 0 0]
    ->  "C\u1ea5m r\u1ebd"
        "C\u1ea5m c\u00f2n l\u1ea1i"
    """
    for dec in decode:
        label_decode = dec
        # print(label_decode)
        for sign in label_decode:

            print(mapping_dict[sign])


def predict_custom(model, imgs, threshold=0.5):
    """
    model: list 
    imgs: np array
    mapping_dict: dict

    result: show annotation
    ->  "C\u1ea5m r\u1ebd"
        "C\u1ea5m c\u00f2n l\u1ea1i"
    """

    if isinstance(imgs, list):
        for img in imgs:
            img = keras.preprocessing.image.img_to_array(img)
            img = np.array([img])
            predictions = model.predict(img)
            predict_decode = np.where(predictions > threshold, 1, 0)
            print('---')
            show_predict_mapping(predict_decode, mapping_dict)

    elif isinstance(imgs, Image):
        img = keras.preprocessing.image.img_to_array(imgs)
        img = np.array([img])
        predictions = model.predict(img)
        predict_decode = np.where(predictions > threshold, 1, 0)
        show_predict_mapping(predict_decode, mapping_dict)
    else:
        print('Error')


def add_value_labels(ax, spacing=3):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
        # positive and negative values.


if __name__ == '__main__':
    base_model_path = Path(OUTPUT_PATH, 'best_model.h5')
    image_val_path = IMAGES_PATH
    image_test_path = IMAGES_PATH

    parser = argparse.ArgumentParser(
        description='Predict the P-glycoprotein Inhibitor.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', type=str, help='model path want to use', default=base_model_path)
    args = parser.parse_args()
    model_path = args.model
    model = keras.models.load_model(model_path)

    print(model.summary())
    # save predict model as image
    keras.utils.plot_model(model, to_file=Path(
        OUTPUT_PATH, 'predict_model.png'), show_shapes=True)

    image_size = model.layers[0].output_shape[0][1:3]
    img = preprocess_input(
        str(Path(image_val_path, 'CHEMBL6.png')), image_size, CHANNELS)  # category: 0
    img_1 = preprocess_input(
        str(Path(image_val_path, 'CHEMBL13.png')), image_size, CHANNELS)  # category: 0
    img_2 = preprocess_input(
        str(Path(image_val_path, 'CHEMBL23.png')), image_size, CHANNELS)  # category: 0
    img_3 = preprocess_input(
        str(Path(image_val_path, 'CHEMBL27.png')), image_size, CHANNELS)  # category: 0
    img_4 = preprocess_input(
        str(Path(image_val_path, 'CHEMBL30.png')), image_size, CHANNELS)  # category: 0
    img_5 = preprocess_input(
        str(Path(image_val_path, 'CHEMBL41.png')), image_size, CHANNELS)  # category: 0
    img_6 = preprocess_input(
        str(Path(image_val_path, 'CHEMBL54.png')), image_size, CHANNELS)  # category: 1
    img_7 = preprocess_input(
        str(Path(image_val_path, 'CHEMBL58.png')), image_size, CHANNELS)  # category: 1
    img_8 = preprocess_input(
        str(Path(image_val_path, 'CHEMBL67.png')), image_size, CHANNELS)  # category: 1

    threshold = 0.5
    predict_custom(model, [img, img_1, img_2, img_3, img_4,
                           img_5, img_6, img_7, img_8], threshold=threshold)
