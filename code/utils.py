"""
Utility functions for file and directory management, date/time helpers, and text processing.
"""

import os
import re
import gc
import shutil
import datetime

import tensorflow as tf
from keras import backend as K


# ========================== Directory Utilities ==========================

def create_directory(directory_path, remove=False):
    """Creates directory. If directory exists and remove=True, removes it first."""
    if remove and os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            os.mkdir(directory_path)
        except Exception:
            print("Could not remove directory: ", directory_path)
            return False
    else:
        try:
            os.mkdir(directory_path)
        except Exception:
            print("Could not create directory: ", directory_path)
            return False
    return True


def remove_directory(directory_path):
    """Removes directory if it exists."""
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
        except Exception:
            print("Could not remove directory: ", directory_path)
            return False
    return True


def clear_directory(directory_path):
    """Removes all files and subdirectories inside the given directory."""
    dirs_files = os.listdir(directory_path)
    for item in dirs_files:
        item_path = directory_path + item
        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(e)
    return True


def remove_empty_folders(path, removeRoot=True):
    """Recursively removes empty folders."""
    if not os.path.isdir(path):
        return
    files = os.listdir(path)
    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                remove_empty_folders(fullpath)
    files = os.listdir(path)
    if len(files) == 0 and removeRoot:
        print("Removing empty folder:", path)
        os.rmdir(path)


def dir_file_count(directory):
    """Returns total number of files in a directory (recursively)."""
    return sum([len(files) for r, d, files in os.walk(directory)])


# ========================== Date/Time Helpers ==========================

def date_time(x):
    """Returns formatted date/time string based on type x."""
    if x == 1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x == 2:
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x == 3:
        return 'Date now: %s' % datetime.datetime.now()
    if x == 4:
        return 'Date today: %s' % datetime.date.today()


# ========================== Text Helpers ==========================

def name_correct(name):
    """Removes everything except alphabetical and selected characters from name string."""
    return re.sub(r'[^a-zA-Z,:]', ' ', name).title()


def debug(x):
    """Prints a debug separator with a value."""
    print("-" * 40, x, "-" * 40)


# ========================== TensorFlow/Keras Reset ==========================

def reset_graph(model=None):
    """Resets TensorFlow graph to free up memory and resource allocation."""
    if model:
        try:
            del model
        except Exception:
            return False
    tf.reset_default_graph()
    K.clear_session()
    gc.collect()
    return True


def reset_callbacks(checkpoint=None, reduce_lr=None, early_stopping=None, tensorboard=None):
    """Resets callback references."""
    checkpoint = None
    reduce_lr = None
    early_stopping = None
    tensorboard = None
