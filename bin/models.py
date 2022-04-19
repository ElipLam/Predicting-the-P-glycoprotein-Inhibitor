import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, MobileNetV2

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(.2, .2),
        layers.RandomContrast(factor=0.2),
    ]
)


def make_model_simple(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(32, 3, strides=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), padding="same")(x)
    x = layers.Dropout(0.25)(x)

    for size in [64, 128, 64, 32]:
        x = layers.Conv2D(size, (3, 3), padding='same')(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(size, (3, 3), padding='same')(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.Activation("relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    activation = "sigmoid"
    units = num_classes

    multi_label_branch = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs=multi_label_branch, name='simple')


def make_model_Keras(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)
    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    activation = "sigmoid"
    units = num_classes

    x = layers.Dropout(0.5)(x)
    multi_label_branch = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs=multi_label_branch, name='simple_Karas')


# # transfer learning
# def make_model_VGG16(input_shape, num_classes):
#     base_model = VGG16(weights="imagenet", include_top=False,
#                        input_shape=input_shape)
#     base_model.trainable = False
#     inputs = keras.Input(shape=input_shape)
#     # Image augmentation block
#     x = data_augmentation(inputs)
#     x = layers.Rescaling(1.0 / 255,)(x)
#     # x = layers.Flatten()(x)

#     # reload tranfer learning model
#     x = base_model(x, training=False)
#     # convert the features to a single 1280-element vector per image.
#     global_average_layer = layers.GlobalAveragePooling2D()
#     # Add a classification head
#     x = global_average_layer(x)
#     x = layers.Dropout(0.2)(x)

#     activation = "sigmoid"
#     units = num_classes
#     outputs = layers.Dense(units, activation=activation)(x)
#     return keras.Model(inputs, outputs, name='VGG16')


# # transfer learning
# def make_model_MobileNetV2(input_shape, num_classes):
#     #  convert color image into a 5x5x1280 block of features
#     base_model = MobileNetV2(
#         weights="imagenet", include_top=False, input_shape=input_shape)
#     base_model.trainable = False

#     inputs = keras.Input(shape=input_shape)
#     x = data_augmentation(inputs)
#     x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)

#     # reload tranfer learning model
#     x = base_model(x, training=False)
#     # convert the features to a single 1280-element vector per image.
#     global_average_layer = layers.GlobalAveragePooling2D()
#     # feature_batch_average = global_average_layer()

#     # Add a classification head
#     x = global_average_layer(x)
#     x = layers.Dropout(0.2)(x)

#     activation = "sigmoid"
#     units = num_classes
#     outputs = layers.Dense(units, activation=activation)(x)
#     return keras.Model(inputs, outputs, name='MobileNetV2')

# # in-house CNN model
# def make_model_DEEPSCREEN_CNNModel_1(input_shape, num_classes):
#     convnet = input_data(shape=input_shape, name='input')

#     convnet = conv_2d(convnet, 32, 5, activation='relu')
#     convnet = max_pool_2d(convnet, 5)

#     convnet = conv_2d(convnet, 64, 5, activation='relu')
#     convnet = max_pool_2d(convnet, 5)

#     convnet = conv_2d(convnet, 128, 5, activation='relu')
#     convnet = max_pool_2d(convnet, 5)

#     convnet = conv_2d(convnet, 64, 5, activation='relu')
#     convnet = max_pool_2d(convnet, 5)

#     convnet = conv_2d(convnet, 32, 5, activation='relu')
#     convnet = max_pool_2d(convnet, 5)

#     convnet = fully_connected(convnet, 1024, activation='relu')
#     convnet = dropout(convnet, 0.8)

#     convnet = fully_connected(convnet, 2, activation='softmax')
#     convnet = regression(convnet, optimizer='adam', learning_rate=0.005,
#                          loss='categorical_crossentropy', name='deepscreen')

#     str_model_name = "{}".format(0.7)

#     model = None

#     # if save_model:
#     #     print("Model will be saved!")
#     #     model = tflearn.DNN(convnet, checkpoint_path='../tflearnModels/{}'.format(str_model_name), best_checkpoint_path='../tflearnModels/bestModels/best_{}'.format(str_model_name),
#     #                         max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir="../tflearnLogs/{}/".format(str_model_name))
#     # else:
#     #     model = tflearn.DNN(convnet)
#     model = tflearn.DNN(convnet)

#     return model


def make_model_DEEPSCREEN_CNNModel(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(32, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(5, padding='same')(x)

    x = layers.Conv2D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(5, padding='same')(x)

    x = layers.Conv2D(128, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(5, padding='same')(x)

    x = layers.Conv2D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(5, padding='same')(x)

    x = layers.Conv2D(32, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(5, padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.8)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs, name='DEEPSCREEN')
