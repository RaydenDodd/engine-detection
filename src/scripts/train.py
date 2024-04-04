from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os
import json
import argparse


def parse_cli():
    parser = argparse.ArgumentParser(description='A tool for extracting MFCCs from audio files')
    parser.add_argument('-m', '--mfccdir', type=str,
                        help='The relative path to the directory that should contain the npz file with the MFCCs', default=None)  # Corrected 'default'
    parser.add_argument('-mf', '--mfcc_filename', type=str,
                        help='The filename of the MFCC you want to train on', default=None)  # Corrected 'default' and minor grammar
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_cli()
    if args.mfccdir is None:
        mfcc_dir = os.path.join('..', 'trained_models')
    else:
        mfcc_dir = args.mfccdir
    if args.mfcc_filename is None:
        mfcc_filename = 'MFCC_all_brands.npz'
    else:
        mfcc_filename = args.mfcc_filename

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    

    all_devices = tf.config.list_physical_devices()
    print("All devices:", all_devices)

    # List all available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs:", gpus)

    # If GPUs are available, set TensorFlow to use the first one
    if gpus:
        try:
            # Set TensorFlow to use only the first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print("Using GPU:", gpus[0])
        except RuntimeError as e:
            # Catch runtime error if modifications are made after program has initialized
            print(e)
    else:
        print("No GPUs found, using CPU instead.")

    # with open(fr'{mfcc_dir}/mfccs.json', "r") as mfcc_file:
    #     data = json.load(mfcc_file)
    # inputs = np.array(data["train"]["mfcc"])
    # targets = np.array(data["train"]["labels"])

    npz_file_path = os.path.join(script_dir, mfcc_dir, mfcc_filename)
    with np.load(npz_file_path) as data:
        inputs = data['features_3d']
        targets = data['labels']
        categorys = data['categorys']

    unique_targets = set(targets)
    unique_categorys = set(categorys)

    print("Unique targets:")
    for target in unique_targets:
        print(target)

    print("\nUnique categories:")
    for category in unique_categorys:
        print(category)

    print("Shape of input:", inputs.shape)
    # Turn the data into train and test sets
    (inputs_train, inputs_test, target_train, target_test) = train_test_split(inputs, targets,
                                                                              test_size=0.2, random_state=42)


    print("Type of inputs_train:", type(inputs_train))
    print("Data type of elements in inputs_train:", inputs_train.dtype)

    print("Type of target_train:", type(target_train))
    print("Data type of elements in target_train:", target_train.dtype)
    # Specify the neural network's architecture
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2])),
        tf.keras.layers.Flatten(),

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
        tf.keras.layers.Dense(len(unique_targets), activation="softmax")

    ])

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
    model.save('engine_detect_rayden_test.keras')
    # model = keras.models.load_model('engine_detect.keras')
