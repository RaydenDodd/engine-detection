from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import matplotlib.pyplot as plt
import os
import json
import argparse
from sklearn.preprocessing import StandardScaler
print(tf.__version__)

def parse_cli():
    parser = argparse.ArgumentParser(description='A script for training a neural network using MFCCs from audio data')
    parser.add_argument('-m', '--mfccdir', type=str,
                        help='The relative path to the directory that should contain the npz file with the MFCCs', default=None)  # Corrected 'default'
    parser.add_argument('-mf', '--mfcc_filename', type=str,
                        help='The filename of the MFCC you want to train on', default=None)  # Corrected 'default' and minor grammar
    return parser.parse_args()

def plot_metrics_and_save(history, model_save_dir):
    metrics = ['loss', 'sparse_categorical_accuracy']
    epochs = range(1, len(history['loss']) + 1)  # Assuming 'loss' is always present
    color = 'blue'  # Default color set to blue

    for metric in metrics:
        plt.figure(figsize=(10, 8))
        
        # Plot training metric
        if metric in history:
            plt.plot(epochs, history[metric], color=color, label='Train')
            # Plot validation metric if it exists
            val_metric_key = 'val_' + metric
            if val_metric_key in history:
                plt.plot(epochs, history[val_metric_key], color=color, linestyle="--", label='Val')

        plt.xlabel('Epoch')
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.title(f'Performance - {metric.replace("_", " ").capitalize()}')
        plt.legend()

        # Save the figure with the metric in the filename
        graph_save_path = os.path.join(model_save_dir, f"{metric}_metrics.png")
        plt.savefig(graph_save_path)
        plt.show()  # Close the figure after saving to avoid display issues

        print(f"Graph for {metric} saved to: {graph_save_path}")

def main(script_dir, mfcc_dir, mfcc_filename):
    all_devices = tf.config.list_physical_devices()
    print("All devices:", all_devices)

    # List all available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs:", gpus)

    # If GPUs are available, set TensorFlow to use the first one
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
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
                                                                              test_size=0.2)

    # NOTE: Adding this scaling seriously hinders the model's performance for some reason
    # With the scaling, the model isn't able to perform even basic binary classification tasks
    # data scaling betwen [0,1]
    #scaler = StandardScaler()
    #inputs_train = scaler.fit_transform(inputs_train.reshape(-1, inputs_train.shape[-1])).reshape(inputs_train.shape)
    #inputs_test = scaler.transform(inputs_test.reshape(-1, inputs_test.shape[-1])).reshape(inputs_test.shape)

    # For CNN
    inputs_train_expanded = np.expand_dims(inputs_train, -1)  # Expanding the last dimension after scaling
    inputs_test_expanded = np.expand_dims(inputs_test, -1)
    print("Shape of inputs_train:", inputs_train_expanded.shape)
    print("Shape of inputs_test:", inputs_test_expanded.shape)
    # Specify the neural network's architecture
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2])),
        tf.keras.layers.Flatten(),

        # First hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        # Second hidden layer
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        # Third hidden layer
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        # Output layer
        tf.keras.layers.Dense(len(unique_targets), activation="softmax")

    ])

    # Compile the network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.000_223_869)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.summary()

    # Train the network
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        patience=5,  # Number of epochs with no improvement after which training will be stopped
        verbose=1,  # Output a message for each early stopping
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )
    history = model.fit(inputs_train, target_train,
              validation_data=(inputs_test, target_test),
              epochs=100,
              batch_size=32,
              callbacks=[early_stopping])

    return history, model


if __name__ == '__main__':
    args = parse_cli()
    if args.mfccdir is None:
        mfcc_dir = os.path.join('..', 'trained_models')
    else:
        mfcc_dir = args.mfccdir
    if args.mfcc_filename is None:
        mfcc_filename = 'MFCC_top_10_brands.npz'
    else:
        mfcc_filename = args.mfcc_filename

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    history, model = main(script_dir, mfcc_dir, mfcc_filename)
    model_path = os.path.join(script_dir, '..', 'trained_models', 'engine_classify.keras')

    history_save_path = os.path.join(script_dir, '..', 'trained_models', f"model_history.json")
    with open(history_save_path, 'w') as f:
        json.dump(history.history, f)
    print(f"Training history at {history_save_path}")

    # Save the model to the specified path
    model.save(model_path)
    # model = keras.models.load_model('engine_detect.keras')

    plot_metrics_and_save(history, model_path)
