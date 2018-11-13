from keras import Model
from keras.applications import inception_v3, imagenet_utils
from keras.layers import GlobalAveragePooling2D, Dense, Dropout


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf')


def RiskFactorModel(n_classes=2, input_shape=(224, 224, 3)):
    base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed10').output)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', name='global_dense1')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='global_dense2')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='softmax', name='global_predictions')(x)
    final_model = Model(inputs=model.input, outputs=predictions)
    return final_model
