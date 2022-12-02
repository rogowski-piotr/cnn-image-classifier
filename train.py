import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input_path", type=str, required=True)
    parser.add_argument("--model_output_path", type=str, required=True)
    parser.add_argument("--dataset_train_path", type=str, required=True)
    parser.add_argument("--val_data_ratio", type=float, required=False, default=0.2)
    parser.add_argument("--epochs_number", type=int, required=False, default=20)
    return parser.parse_args()


def prepare_data_generator(val_data_ratio):
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=val_data_ratio
    )


def prepare_train_dataset(data_generator, dataset_path):
    return data_generator.flow_from_directory(
        directory=dataset_path,
        target_size=(50,50),
        color_mode='grayscale',
        batch_size=20,
        class_mode='binary',
        subset='training'
    )


def prepare_validation_dataset(data_generator, dataset_path):
    return data_generator.flow_from_directory(
        directory=dataset_path,
        target_size=(50,50),
        color_mode='grayscale',
        batch_size=20,
        class_mode='binary',
        subset='validation'
    )


def train(model, epochs_num, dataset_path, val_data_ratio):
    data_generator = prepare_data_generator(val_data_ratio)
    train_generator = prepare_train_dataset(data_generator, dataset_path)
    val_generator = prepare_validation_dataset(data_generator, dataset_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit_generator(train_generator, epochs=epochs_num, validation_data=val_generator, validation_steps=50)
    plot_epoch(history)
    return model


def plot_epoch(train_history):
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.plot(train_history.history['accuracy'])
    plt.plot(train_history.history['val_accuracy'])
    plt.title('Loss and Accuracy per Epoch')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Validation Loss', 'Train Accuracy', 'Validation Accuracy'], loc='upper left')
    plt.savefig('train.png')
    plt.close()


if __name__ == '__main__':
    args = parse_args()
    model = keras.models.load_model(args.model_input_path)
    model = train(
        model=model,
        epochs_num=args.epochs_number,
        dataset_path=args.dataset_train_path,
        val_data_ratio=args.val_data_ratio
    )
    model.save(args.model_output_path)
