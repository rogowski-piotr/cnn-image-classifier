import argparse
import tensorflow as tf
from tensorflow.keras.layers import Dropout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_path", type=str, required=False, default="./models/initial_model.h5")
    return parser.parse_args()


def visualize_model(model):
    tf.keras.utils.plot_model(
        model, to_file="./plots/model.png", 
        show_shapes=True, 
        show_layer_activations=True, 
        show_dtype=True,
        show_layer_names=True
    )


def generate_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=(50, 50, 1), padding='same'))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    visualize_model(model)
    return model


if __name__ == '__main__':
    args = parse_args()
    model = generate_model()
    model.save(args.model_output_path)
