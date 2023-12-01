from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import json
import argparse


def parse_cli():
    parser = argparse.ArgumentParser(description='A tool for extracting MFCCs from audio files')
    parser.add_argument('-m', '--mfccdir', type=str,
                        help='The relative path to the directory that should contain folders with mfccs extracted from the chunked audio files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_cli()
    mfcc_dir = args.mfccdir

    tf.config.set_visible_devices([], 'GPU')

    with open(fr'{mfcc_dir}/mfccs.json', "r") as mfcc_file:
        data = json.load(mfcc_file)
    inputs = np.array(data["train"]["mfcc"])
    targets = np.array(data["train"]["labels"])

    # Turn the data into train and test sets
    (inputs_train, inputs_test, target_train, target_test) = train_test_split(inputs, targets,
                                                                              test_size=0.2)

    # Specify the neural network's architecture
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # First hidden layer
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        # Second hidden layer
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        # Third hidden layer
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        # Output layer
        tf.keras.layers.Dense(4, activation="softmax")])

    # Compile the network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.000_1)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Train the network
    model.fit(inputs_train, target_train,
              validation_data=(inputs_test, target_test),
              epochs=100,
              batch_size=32)

    # Save the model
    model.save('engine_detect.keras')
    # model = keras.models.load_model('engine_detect.keras')
