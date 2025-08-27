import argparse, json, os, random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_model(input_shape=(28, 28, 1), num_classes=10):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            data_augmentation,
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def plot_confusion_matrix(cm, class_names, out_path):
    fig = plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (normalized)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out", type=str, default="fashion_model.keras")
    args = parser.parse_args()

    set_seeds(42)

    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    # Split validation from train
    val_split = 0.1
    num_val = int(len(x_train) * val_split)
    x_val = x_train[:num_val]
    y_val = y_train[:num_val]
    x_train_small = x_train[num_val:]
    y_train_small = y_train[num_val:]

    model = build_model()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(args.out, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        x_train_small, y_train_small,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # Save training history
    with open("training_history.json", "w") as f:
        json.dump(history.history, f)

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    # Classification report + confusion matrix
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    plot_confusion_matrix(cm, class_names, "confusion_matrix.png")

if __name__ == "__main__":
    main()
