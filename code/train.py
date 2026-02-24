"""
Training pipeline for Pneumonia detection model.
Handles data loading, augmentation, model training, and evaluation.

Usage:
    python train.py
"""

import os
import time
import random

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

from utils import (create_directory, clear_directory, remove_empty_folders,
                   dir_file_count, date_time, name_correct, reset_graph, reset_callbacks)
from model import get_conv_model, get_inception_model


# ========================== Configuration ==========================

INPUT_DIR = r"data/input/"
OUTPUT_DIR = r"data/output/"

TRAINING_DIR = INPUT_DIR + r"train"
VALIDATION_DIR = INPUT_DIR + r"val"
TESTING_DIR = INPUT_DIR + r"test"

FIGURE_DIR = OUTPUT_DIR + "figures"

RESCALE = 1.0 / 255
TARGET_SIZE = (150, 150)
BATCH_SIZE = 163
CLASS_MODE = "categorical"
EPOCHS = 100


def setup_directories():
    """Creates required output directories."""
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR, exist_ok=True)


def get_data_generators():
    """Creates and returns train, validation, and test data generators."""
    train_datagen = ImageDataGenerator(
        rescale=RESCALE,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR, target_size=TARGET_SIZE,
        class_mode=CLASS_MODE, batch_size=BATCH_SIZE, shuffle=True
    )

    validation_datagen = ImageDataGenerator(rescale=RESCALE)
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR, target_size=TARGET_SIZE,
        class_mode=CLASS_MODE, batch_size=dir_file_count(VALIDATION_DIR), shuffle=False
    )

    test_datagen = ImageDataGenerator(rescale=RESCALE)
    test_generator = test_datagen.flow_from_directory(
        TESTING_DIR, target_size=TARGET_SIZE,
        class_mode=CLASS_MODE, batch_size=dir_file_count(TESTING_DIR), shuffle=False
    )

    return train_generator, validation_generator, test_generator


def get_class_weights(train_generator):
    """Computes balanced class weights from training data."""
    return class_weight.compute_class_weight(
        'balanced', np.unique(train_generator.classes), train_generator.classes
    )


def setup_model_dirs():
    """Creates timestamped model and log directories. Returns model file path."""
    main_model_dir = OUTPUT_DIR + r"models/"
    main_log_dir = OUTPUT_DIR + r"logs/"

    clear_directory(main_log_dir)
    remove_empty_folders(main_model_dir, False)

    timestamp = time.strftime('%Y-%m-%d %H-%M-%S')
    model_dir = main_model_dir + timestamp + "/"
    log_dir = main_log_dir + timestamp

    create_directory(model_dir, remove=True)
    create_directory(log_dir, remove=True)

    model_file = model_dir + "{epoch:02d}-val_acc-{val_acc:.2f}-val_loss-{val_loss:.2f}.hdf5"
    return model_file, log_dir


def get_callbacks(model_file, log_dir):
    """Creates and returns training callbacks."""
    checkpoint = ModelCheckpoint(model_file, monitor='val_acc', save_best_only=True)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, verbose=1, restore_best_weights=True
    )

    tensorboard = TensorBoard(
        log_dir=log_dir, batch_size=BATCH_SIZE, update_freq='batch'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', patience=5, cooldown=2, min_lr=1e-10, verbose=1
    )

    return [checkpoint, reduce_lr, early_stopping, tensorboard]


def train_model(model, train_generator, validation_generator, callbacks, class_weights):
    """Compiles and trains the model. Returns training history."""
    print("Starting Training Model", date_time(1))

    optimizer = optimizers.Adam()
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        verbose=2,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        class_weight=class_weights
    )

    print("Completed Model Training", date_time(1))
    return history


def evaluate_model(model, test_generator, classes):
    """Evaluates the model on test data and prints metrics."""
    print("Evaluating model...")
    result = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=1)
    print("Loss     : %.2f" % result[0])
    print("Accuracy : %.2f%%" % (result[1] * 100))

    y_pred = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
    y_pred = y_pred.argmax(axis=-1)
    y_true = test_generator.classes

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("-" * 60)
    print("Precision : %.2f%%" % (precision * 100))
    print("Recall    : %.2f%%" % (recall * 100))
    print("F1-Score  : %.2f%%" % (f1 * 100))
    print("-" * 60)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    return y_true, y_pred


# ========================== Main ==========================

if __name__ == "__main__":
    reset_graph()
    reset_callbacks()

    setup_directories()

    # Load data
    train_gen, val_gen, test_gen = get_data_generators()
    cw = get_class_weights(train_gen)

    classes = os.listdir(TRAINING_DIR)
    classes = [name_correct(c) for c in classes]

    # Setup model directories and callbacks
    model_file, log_dir = setup_model_dirs()
    callbacks = get_callbacks(model_file, log_dir)

    # Build model (switch to get_inception_model() for transfer learning)
    reset_graph()
    reset_callbacks()
    print("Getting Base Model", date_time(1))
    model = get_conv_model()

    # Train
    history = train_model(model, train_gen, val_gen, callbacks, cw)

    # Evaluate
    y_true, y_pred = evaluate_model(model, test_gen, classes)

    print("\nTraining complete.")
