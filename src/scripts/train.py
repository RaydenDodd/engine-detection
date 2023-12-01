from sklearn.model_selection import train_test_split
import tensorflow as tf

import json
import numpy as np
import argparse
from sklearn.metrics import accuracy_score  # load and convert data
import matplotlib.pyplot as plt

def parse_cli():
    parser = argparse.ArgumentParser(description='A tool for extracting MFCCs from audio files')
    parser.add_argument('-m', '--mfccdir', type=str, help='The relative path to the directory that should contain folders with mfccs extracted from the chunked audio files')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_cli()
    mfcc_dir = args.mfccdir

    tf.config.set_visible_devices([], 'GPU')

    with open(fr'{mfcc_dir}/mfccs.json', "r") as fp:
        data = json.load(fp)
    inputs = np.array(data["train"]["mfcc"])
    targets = np.array(data["train"]["labels"])

    # turn data into train and testset
    (inputs_train, inputs_test, target_train, target_test) = train_test_split(inputs, targets,
                                                                              test_size=0.2)  # build the network architecture

    model = tf.keras.Sequential([
        # input layer
        tf.keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # 1st hidden layer
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        # 2nd hidden layer
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        # 3rd hidden layer
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        # output layer
        tf.keras.layers.Dense(4, activation="softmax")])

    # Compile the network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.000_1)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()  # train the network
    history = model.fit(inputs_train, target_train,
                        validation_data=(inputs_test, target_test),
                        epochs=100,
                        batch_size=32)

    # Save the model
    model.save('engine_detect.keras')

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))  # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train_accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test_accuracy")
    axs[0].set_ylabel("Accuracy", fontsize=18)
    axs[0].legend(loc="lower right", prop={"size": 16})
    axs[0].set_title("Accuracy evaluation", fontsize=20)
    axs[0].tick_params(axis="both", labelsize=16)  # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error", fontsize=18)
    axs[1].legend(loc="upper right", prop={"size": 16})
    axs[1].set_title("Error evaluation", fontsize=20)
    axs[1].tick_params(axis="both", labelsize=16)
    fig.savefig("accuracy_error.png",
                bbox_inches="tight")
    plt.show()
