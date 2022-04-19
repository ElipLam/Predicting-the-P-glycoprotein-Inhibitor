import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from data_preprocessing import str_list, preprocessing_image_pipeline, preprocessing
from models import make_model_DEEPSCREEN_CNNModel, make_model_Keras, make_model_simple
from data_preprocessing import CHANNELS

ROOT = Path(__file__).resolve().parents[1]
BIN_PATH = Path(ROOT, 'bin')
OUTPUT_PATH = Path(ROOT, 'output')
DATASET_PATH = Path(ROOT, 'dataset')
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 128
NUM_CLASSES = 1
EPOCHS = 2
MODEL = 'simple'  # keras / simple / vgg16 / mobilenetv2


def load_dataset():
    dataset_filename = 'P-gp_act_inact_dataset.csv'
    # create dataset
    dataset_csv = os.path.join(
        DATASET_PATH, dataset_filename)

    labels = []
    images_path = []

    if not os.path.isfile(dataset_csv):
        print(f'{dataset_filename} does not exist!')
        print(f'Creating dataset, please wait a moment...')
        preprocessing()
    else:
        print(f'{dataset_filename} already exists!')
        print('Loading dataset...')
    df = pd.read_csv(dataset_csv)
    images_path = df['Path'].to_list()
    labels = df['Activity'].to_list()
    # labels = [*map(str_list, labels)]
    print('Done!')
    return images_path, labels


def training(model_name=MODEL, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, epochs=EPOCHS):
    lb_encoding = LabelEncoder()
    # convert to numpy array
    images_path, labels = load_dataset()
    X = np.array(images_path)
    Y = lb_encoding.fit_transform(labels)

    model = keras.Model()
    if model_name == 'keras':
        model = make_model_Keras(input_shape=image_size +
                                 (CHANNELS,), num_classes=NUM_CLASSES)
    # elif model_name == 'vgg16':
    #     model = make_model_VGG16(input_shape=image_size +
    #                              (CHANNELS,), num_classes=NUM_CLASSES)
    # elif model_name == 'mobilenetv2':
    #     model = make_model_MobileNetV2(
    #         input_shape=image_size + (CHANNELS,), num_classes=NUM_CLASSES)
    elif model_name == 'deepscreen':
        model = make_model_DEEPSCREEN_CNNModel(
            input_shape=image_size + (CHANNELS,), num_classes=NUM_CLASSES)
    else:
        model = make_model_simple(
            input_shape=image_size + (CHANNELS,), num_classes=NUM_CLASSES)

    model.compile(
        optimizer=keras.optimizers.SGD(),
        loss="binary_crossentropy",
        metrics=['accuracy', tf.keras.metrics.Recall(),
                 tf.keras.metrics.Precision()],
    )

    # save training model as image
    keras.utils.plot_model(model, to_file=Path(
        OUTPUT_PATH, 'train_model.png'), show_shapes=True)
    model.summary()
    kf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
    histories = {}
    histories['accuracy'] = []
    histories['val_accuracy'] = []
    histories['loss'] = []
    histories['val_loss'] = []

    print(
        f'Training {model_name} model with batch size = {batch_size}, image size = {image_size} and epochs = {epochs}.')
    for epo in range(1, epochs+1):
        histories[epo] = {}
        histories[epo]['accuracy'] = []
        histories[epo]['val_accuracy'] = []
        histories[epo]['loss'] = []
        histories[epo]['val_loss'] = []
        print(f'Epochs {epo}/{epochs}')

        fold_indx = 1
        for train_index, val_index in kf.split(X, Y):
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    Path(OUTPUT_PATH, f"save_at_{epo}_{fold_indx}.h5")),
            ]
            print('Training Fold', fold_indx)
            train_X = X[train_index]
            train_Y = Y[train_index]
            val_X = X[val_index]
            val_Y = Y[val_index]
            # print('Observations:', train_X.shape[0])
            # print('Number of Activity:', train_Y.sum())

            main_preprocessing_pipeline = preprocessing_image_pipeline(
                image_size)
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_X, train_Y))
            train_dataset = (train_dataset
                             .map(main_preprocessing_pipeline, num_parallel_calls=tf.data.AUTOTUNE)
                             .batch(batch_size)
                             .prefetch(tf.data.AUTOTUNE))

            val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_Y))
            val_dataset = (val_dataset
                           .map(main_preprocessing_pipeline, num_parallel_calls=tf.data.AUTOTUNE)
                           .batch(batch_size)
                           .prefetch(tf.data.AUTOTUNE))
            history = model.fit(train_dataset, steps_per_epoch=len(train_dataset),
                                validation_data=val_dataset, validation_steps=len(
                                    val_dataset),
                                callbacks=callbacks, epochs=1)

            histories[epo]['accuracy'] += history.history['accuracy']
            histories[epo]['val_accuracy'] += history.history['val_accuracy']
            histories[epo]['loss'] += history.history['loss']
            histories[epo]['val_loss'] += history.history['val_loss']

            fold_indx += 1

        # mean eachs epochs
        histories[epo]['mean_accuracy'] = np.mean(histories[epo]['accuracy'])
        histories[epo]['mean_val_accuracy'] = np.mean(
            histories[epo]['val_accuracy'])
        histories[epo]['mean_loss'] = np.mean(histories[epo]['loss'])
        histories[epo]['mean_val_loss'] = np.mean(histories[epo]['val_loss'])

        histories['accuracy'].append(histories[epo]['mean_accuracy'])
        histories['val_accuracy'].append(histories[epo]['mean_val_accuracy'])
        histories['loss'].append(histories[epo]['mean_loss'])
        histories['val_loss'].append(histories[epo]['mean_val_loss'])
    return histories


def plot_learning_curves(histories):
    acc = histories['accuracy']
    val_acc = histories['val_accuracy']

    loss = histories['loss']
    val_loss = histories['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xticks(range(len(val_loss)), labels=range(1, len(val_loss)+1))
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), max(plt.ylim())])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xticks(range(len(val_loss)), labels=range(1, len(val_loss)+1))
    plt.ylabel('Cross Entropy')
    plt.ylim([min(plt.ylim()), max(plt.ylim())])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(Path(ROOT, 'output', 'training_accuracy.png'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the P-glycoprotein Inhibitor.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bs', '--batch_size', type=int,
                        help='batch size', default=128)
    parser.add_argument('--isize', '--image_size', nargs=2,
                        type=int, help='image size', default=(224, 224))
    parser.add_argument('--epos', '--epochs', type=int,
                        help='epochs', default=2)
    parser.add_argument(
        '--model', type=str, help='model name', choices=['keras', 'deepscreen', 'simple'], default='deepscreen')
    args = parser.parse_args()

    model_name = args.model
    bs = args.bs
    isize = tuple(args.isize)
    epos = args.epos
    # model_name = keras / simple / vgg16 / mobilenetv2
    his = training(model_name=model_name, batch_size=bs,
                   image_size=isize, epochs=epos)
    with open(Path(OUTPUT_PATH, 'acc_loss_history.json'), 'w') as outfile:
        json.dump(his, outfile, indent=4)
    plot_learning_curves(his)

""" 
dataset structure:
    {
        id: {
            # data : [],
            labels: [],
            height: int
            width: int,
            path: str,
        },
    }

"images": [
        {
            "file_name": "*.png",
            "height": int,
            "width": int,
            "id": int,
            "street_id": int
        },
        ]

"annotations": [
        {
            "segmentation": [],
            "area": int,
            "iscrowd": int,
            "image_id": int,
            "bbox": [
                int,
                int,
                int,
                int
            ],
            "category_id": int,
            "id": int
        },
        ]

histories structure:
    {
        epoch:{
                accuracy : [kfold],
                val_accuracy: [kfold],
                loss: [kfold]
                val_loss: [kfold],
                mean_accuracy: float
                mean_val_accuracy: float
                mean_loss: float
                mean_val_loss: float
            },
        accuracy: [epoch mean_accuracy] 
        val_accuracy: [epoch mean_val_accuracy]
        loss: [epoch mean_loss] 
        val_loss: [epoch mean_val_loss] 
    }  
"""
