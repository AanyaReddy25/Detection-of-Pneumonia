"""
Visualization functions for data exploration and prediction display.
"""

import os
import random

import cv2
import seaborn as sns
import matplotlib.pyplot as plt

from utils import name_correct


# ========================== Subplot/Plot Parameter Setup ==========================

def get_reset_subplot_params(nrows, ncols, dpi):
    """Returns a dictionary of subplot configuration parameters."""
    subplot_params = {
        "nrows": nrows,
        "ncols": ncols,
        "figsize_col": ncols * 2.5,
        "figsize_row": nrows * 2.5,
        "dpi": dpi,
        "facecolor": 'w',
        "edgecolor": 'k',
        "subplot_kw": {'xticks': [], 'yticks': []},
        "axes.titlesize": 'small',
        "hspace": 0.5,
        "wspace": 0.3,
    }
    return subplot_params


def get_reset_plot_params(figsize=(15, 5), title="", xlabel="", ylabel="",
                          legends=[], title_fontsize=18, label_fontsize=14,
                          image_file_name="", save=False, dpi=100, update_image=True):
    """Returns a dictionary of plot configuration parameters."""
    plot_params = {
        "figsize": figsize,
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "legends": legends,
        "title_fontsize": title_fontsize,
        "axes.titlesize": "small",
        "label_fontsize": label_fontsize,
        "image_file_name": image_file_name,
        "save": save,
        "update_image": update_image,
        "subplot": None,
    }
    return plot_params


# ========================== Image Sampling ==========================

def select_image_by_category(image_dir, image_count_per_category):
    """Selects random images from each class subdirectory."""
    classes = os.listdir(image_dir)
    class_count = len(classes)
    image_file_paths = {}

    for i in range(class_count):
        subdir_path = image_dir + "/" + classes[i]
        subdir_files = os.listdir(subdir_path)
        subdir_file_count = len(subdir_files)
        subdir_file_mem = {}
        subdir_file_index = -1
        image_file_paths[classes[i]] = []

        for j in range(image_count_per_category):
            while subdir_file_index in subdir_file_mem:
                subdir_file_index = random.randint(0, subdir_file_count - 1)
            subdir_file_mem[subdir_file_index] = 1
            subdir_file_name = subdir_files[subdir_file_index]
            subdir_file_path = subdir_path + "/" + subdir_file_name
            image_file_paths[classes[i]].append(subdir_file_path)

    return image_file_paths


# ========================== Plotting Functions ==========================

def get_fig_axs(subplot_params):
    """Creates matplotlib figure and axes from subplot parameters."""
    fig, axs = plt.subplots(
        nrows=subplot_params["nrows"], ncols=subplot_params["ncols"],
        figsize=(subplot_params["figsize_col"], subplot_params["figsize_row"]),
        dpi=subplot_params["dpi"], facecolor=subplot_params["facecolor"],
        edgecolor=subplot_params["edgecolor"], subplot_kw=subplot_params["subplot_kw"]
    )
    return fig, axs


def plot_sample_image(image_file_paths, plot_params, subplot_params, update_image=True):
    """Plots a grid of sample images."""
    fig, axs = get_fig_axs(subplot_params)
    plt.rcParams.update({'axes.titlesize': plot_params["axes.titlesize"]})
    plt.subplots_adjust(hspace=subplot_params["hspace"], wspace=subplot_params["wspace"])

    for i, img_filepath in enumerate(image_file_paths):
        img = cv2.imread(img_filepath, 1)
        plt.title(img_filepath.split("/")[-1])
        plt.subplot(subplot_params["nrows"], subplot_params["ncols"], i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    if plot_params["update_image"] and os.path.exists(plot_params["image_file_name"]):
        os.remove(plot_params["image_file_name"])
    if plot_params["save"]:
        fig.savefig(plot_params["image_file_name"], dpi=plot_params["dpi"])

    plt.tight_layout()
    plt.show()


def show_class_sample_images(directory, image_count_per_category=5, save=False, dpi=100, update_image=False):
    """Displays random sample images from each class in the directory."""
    class_count = len(os.listdir(directory))
    print("Number of Class: ", class_count)
    sample_img_by_class = select_image_by_category(directory, image_count_per_category)
    for class_name in sample_img_by_class:
        plot_params = get_reset_plot_params(image_file_name="img.png", save=save, dpi=dpi, update_image=update_image)
        subplot_params = get_reset_subplot_params(nrows=1, ncols=image_count_per_category, dpi=dpi)
        print("%s%s%s" % ("-" * 55, name_correct(class_name), "-" * 55))
        plot_sample_image(sample_img_by_class[class_name], plot_params, subplot_params)
        print("")
    print("%s%s%d%s" % ("-" * 55, "All Class Printed:", class_count, "-" * 55))


# ========================== Bar Plot Functions ==========================

def subdirectory_file_count(master_directory):
    """Counts number of files in each subdirectory of a directory."""
    subdirectories = os.listdir(master_directory)
    subdirectory_names = []
    subdirectory_file_counts = []

    for subdirectory in subdirectories:
        current_directory = os.path.join(master_directory, subdirectory)
        file_count = len(os.listdir(current_directory))
        subdirectory_names.append(subdirectory)
        subdirectory_file_counts.append(file_count)

    return subdirectory_names, subdirectory_file_counts


def bar_plot(x, y, plot_property):
    """Displays a bar plot using seaborn."""
    if plot_property['subplot']:
        plt.subplot(plot_property['subplot'])
    sns.barplot(x=x, y=y)
    plt.title(plot_property['title'], fontsize=plot_property['title_fontsize'])
    plt.xlabel(plot_property['xlabel'], fontsize=plot_property['label_fontsize'])
    plt.ylabel(plot_property['ylabel'], fontsize=plot_property['label_fontsize'])
    plt.xticks(range(len(x)), x)


def count_bar_plot(master_directory, plot_property):
    """Shows bar plot for count of labels in subdirectory of a directory."""
    dir_name, dir_file_count = subdirectory_file_count(master_directory)
    x = [name_correct(i) for i in dir_name]
    y = dir_file_count
    bar_plot(x, y, plot_property)


def show_train_val_test(training_dir, validation_dir, testing_dir, plot_property):
    """Shows bar plots for training, validation, and testing data distribution."""
    plt.figure(figsize=plot_property['figsize'])

    title = plot_property['title']
    plot_property['title'] = title + " (Training)"
    subplot_no = plot_property['subplot']
    count_bar_plot(training_dir, plot_property)

    plot_property['title'] = title + " (Validation)"
    plot_property['subplot'] = subplot_no + 1
    count_bar_plot(validation_dir, plot_property)

    plot_property['title'] = title + " (Testing)"
    plot_property['subplot'] = subplot_no + 2
    count_bar_plot(testing_dir, plot_property)

    plt.show()


# ========================== Prediction Visualization ==========================

def show_predictions(y_img_batch, y_true, y_pred, subplot_params, plot_params,
                     class_map, testing_dir, image_file_name, test_generator,
                     count=8, sample=True):
    """Visualizes model predictions vs. true labels on test images."""
    fig, axs = get_fig_axs(subplot_params)
    plt.rcParams.update({'axes.titlesize': plot_params["axes.titlesize"]})
    plt.subplots_adjust(hspace=subplot_params["hspace"], wspace=subplot_params["wspace"])

    file_names = test_generator.filenames
    m = {}
    length = len(y_true)

    for i in range(0, count):
        num = i
        if sample:
            num = random.randint(0, length - 1)
            while num in m:
                num = int(random.randint(0, length - 1))
            m[num] = 1

        plt.subplot(subplot_params["nrows"], subplot_params["ncols"], i + 1)
        img = cv2.imread(testing_dir + "\\" + file_names[num], 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

        original = class_map[y_true[num]]
        predicted = class_map[y_pred[num]]

        title_text = "True: %s\nPred: %s" % (original, predicted)

        if original == predicted:
            plt.title(title_text)
        else:
            plt.title(title_text, color='red')

        if plot_params["update_image"] and os.path.exists(image_file_name):
            os.remove(image_file_name)
        fig.savefig(image_file_name, dpi=subplot_params["dpi"])

    plt.tight_layout()
    plt.show()
