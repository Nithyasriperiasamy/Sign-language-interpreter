"""
train.py
--------
Trains a Convolutional Neural Network (CNN) to classify American Sign
Language (ASL) alphabet hand gestures (A-Z, space, delete, nothing).

Usage:
    python train.py --data-dir dataset/asl_alphabet_train --epochs 30

The trained model is saved to models/cnn_model.keras, and training/
validation accuracy & loss curves plus a confusion matrix are saved
as PNG files under models/.

Dataset:
    This script expects the Kaggle "ASL Alphabet" dataset, organized as:

        dataset/
            asl_alphabet_train/
                A/
                B/
                ...
                Z/
                space/
                del/
                nothing/

    Download instructions are printed automatically if the dataset
    directory is missing (see `print_dataset_instructions`).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend, safe for headless training
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.preprocessing import IMG_SIZE

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

MODELS_DIR = Path("models")
DEFAULT_DATA_DIR = Path("dataset/asl_alphabet_train")
BATCH_SIZE = 32
NUM_CLASSES = 29  # 26 letters + space + del + nothing


def print_dataset_instructions() -> None:
    """Print instructions for downloading and organizing the ASL dataset."""
    print(
        """
        ============================================================
         ASL Alphabet dataset not found!
        ============================================================
        Please download it from Kaggle:

            https://www.kaggle.com/datasets/grassknoted/asl-alphabet

        After downloading and extracting, organize it like this:

            dataset/
                asl_alphabet_train/
                    A/  B/  C/  ...  Z/
                    space/
                    del/
                    nothing/

        You can download it via the Kaggle CLI:

            pip install kaggle
            kaggle datasets download -d grassknoted/asl-alphabet
            unzip asl-alphabet.zip -d dataset/

        Then re-run:

            python train.py --data-dir dataset/asl_alphabet_train
        ============================================================
        """
    )


def build_model(input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Build and compile the CNN architecture.

    Architecture (as specified):
        Conv2D -> ReLU -> MaxPooling
        Conv2D -> ReLU -> MaxPooling
        Conv2D -> ReLU
        Flatten -> Dense -> Dropout -> Dense(Softmax)

    Args:
        input_shape: Shape of a single input image, e.g. (64, 64, 3).
        num_classes: Number of output classes.

    Returns:
        A compiled tf.keras.Model ready for training.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), padding="same"),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.Activation("relu"),
            layers.Flatten(),
            layers.Dense(256),
            layers.Activation("relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="asl_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_data_generators(
    data_dir: Path, img_size: tuple[int, int], batch_size: int, validation_split: float = 0.2
):
    """Create training and validation data generators with augmentation.

    Args:
        data_dir: Path to the directory containing one subfolder per class.
        img_size: Target (width, height) for resizing images.
        batch_size: Number of images per training batch.
        validation_split: Fraction of data reserved for validation.

    Returns:
        A tuple (train_generator, val_generator).
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.8, 1.2),
        horizontal_flip=False,  # ASL letters are handedness-sensitive
        validation_split=validation_split,
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_generator, val_generator


def plot_training_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    """Plot and save training/validation accuracy and loss curves.

    Args:
        history: The History object returned by model.fit().
        output_dir: Directory in which to save the plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_plot.png")
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_plot.png")
    plt.close()

    print(f"Saved accuracy_plot.png and loss_plot.png to {output_dir}/")


def plot_confusion_matrix(model: tf.keras.Model, val_generator, class_names: list[str], output_dir: Path) -> None:
    """Compute and save a confusion matrix heatmap for the validation set.

    Args:
        model: Trained Keras model.
        val_generator: Validation data generator (shuffle must be False).
        class_names: List of class labels in index order.
        output_dir: Directory in which to save the plot.
    """
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes[: len(y_pred)]

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    print(f"Saved confusion_matrix.png to {output_dir}/")


def main() -> None:
    """Entry point: parse arguments, train the model, and save all artifacts."""
    parser = argparse.ArgumentParser(description="Train the ASL CNN classifier.")
    parser.add_argument(
        "--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to training data directory."
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print_dataset_instructions()
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building data generators...")
    train_generator, val_generator = build_data_generators(data_dir, IMG_SIZE, args.batch_size)

    num_classes = train_generator.num_classes
    class_indices = train_generator.class_indices
    class_names = sorted(class_indices, key=lambda k: class_indices[k])
    print(f"Detected {num_classes} classes: {class_names}")

    print("Building model...")
    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)
    model.summary()

    model_path = MODELS_DIR / "cnn_model.keras"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Ensure the final best model is saved (ModelCheckpoint already does this,
    # but we save again in case training finished without an improvement).
    if not model_path.exists():
        model.save(model_path)

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"\nFinal Training Accuracy:   {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")

    plot_training_history(history, MODELS_DIR)
    plot_confusion_matrix(model, val_generator, class_names, MODELS_DIR)

    # Save class names alongside the model so predict.py doesn't need to
    # re-derive them from the dataset directory at inference time.
    with open(MODELS_DIR / "class_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    print(f"\nModel saved to: {model_path}")
    print(f"Class names saved to: {MODELS_DIR / 'class_names.txt'}")


if __name__ == "__main__":
    main()
