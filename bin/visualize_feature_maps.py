from pathlib import Path
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from data_preprocessing import CHANNELS, preprocess_input

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = Path(ROOT, 'output')
DATASET_PATH = Path(ROOT, 'dataset')
IMAGES_PATH = Path(DATASET_PATH, 'images')
model_path = Path(OUTPUT_PATH, 'best_model.h5')

# model = keras.models.load_model(model_path)


if __name__ == '__main__':
    model = keras.models.load_model(model_path)
    # model.summary()

    for i, layer in enumerate(model.layers):
        if 'conv' not in layer.name:
            continue
        filters, bias = layer.get_weights()
        print(i, layer.name, filters.shape)
        # print(bias)

    # redifine model to output right after the first hidden layers
    blocks = [8, 11, 16, 19, 27]
    outputs = [model.layers[i].output for i in blocks]
    sub_model = keras.models.Model(
        inputs=model.inputs, outputs=outputs)
    sub_model.summary()
    # load image with the required shape
    img = preprocess_input(
        str(Path(IMAGES_PATH, 'CHEMBL6.png')), target_size=(224, 224), channels=CHANNELS)

    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = np.expand_dims(img, axis=0)
    # prepare th image (e.g. scale pixel values for the vgg)
    # img = preprocess_input(img)
    feature_maps = sub_model.predict(img)
    # plot all 64 maps in an 8x8 squares
    h_fig = 8
    w_fig = 8
    for i, fmap in zip(blocks, feature_maps):
        ix = 1
        fig = plt.figure(figsize=(7, 5))
        block_name = 'BLOCK_{}'.format(i)
        fig.suptitle(block_name)
        # specify subplot and return of axis
        print(block_name, fmap.shape[3], 'filters, creating image...')
        for j in range(1, (h_fig * w_fig)+1):
            ax = plt.subplot(h_fig, w_fig, j)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot fiter channel in grayscale
            plt.imshow(fmap[0, :, :, j-1], cmap='gray')
            # save figure
            plt.savefig('{}.png'.format(
                Path(OUTPUT_PATH, 'images', block_name)))
        # print(fmap)
        ix += 1
    plt.show()
