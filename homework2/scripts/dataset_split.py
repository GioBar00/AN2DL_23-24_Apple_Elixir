import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

training_data = np.load("../dataset/training_data_clean.npy", allow_pickle=True)
categories = np.load("../dataset/categories_clean.npy", allow_pickle=True)
series_lengths = np.load("../dataset/series_length_clean.npy", allow_pickle=True)

global num_windows

def build_sequence(data, categories, window_size=200, stride=200, telescope=18):
    global num_windows
    num_windows = 0
    actual_window = window_size + telescope

    new_categories = []
    # two dimensional array (number of series, window size)
    X = []
    # two dimensional array (number of series, telescope
    y = []

    for i, element in enumerate(data):
        length = len(element)
        new_stride = stride
        # number of windows
        n_windows = int(np.ceil((length - actual_window) / new_stride)) + 1
        if n_windows < 1:
            n_windows = 1
        # print("Length: ", length, " n_windows: ", n_windows)
        # reevaluate the stride
        if n_windows > 1:
            new_stride = int((length - actual_window) / (n_windows - 1)) + 1
        num_windows += n_windows

        # start from the end of the series
        start_idx = length - actual_window
        end_idx = length
        # for each window
        for j in range(n_windows):
            if start_idx < 0:
                start_idx = 0
                end_idx = actual_window
                if end_idx > length:
                    end_idx = length
            # append the window to X
            temp = element[start_idx:end_idx - telescope]
            if len(temp) < window_size:
                temp = np.pad(temp, (0, window_size - len(temp)), mode="constant", constant_values=-1)
            X.append(temp)
            # append the telescope to y
            y.append(element[end_idx - telescope:end_idx])
            # append the category
            new_categories.append(categories[i])
            # update the start and end index
            start_idx -= new_stride
            end_idx -= new_stride

    return X, y, new_categories


def plot_window_telescope(window, telescope, predicted_telescope=None):
    # remove padding if present
    window = window[window != -1.]
    # connect window and telescope
    telescope = np.concatenate((window[-1:], telescope))
    # plot window and telescope
    plt.plot(np.arange(len(window)), window, label="window")
    plt.plot(np.arange(len(window) - 1, len(window) - 1 + len(telescope)), telescope, label="telescope")
    if predicted_telescope is not None:
        predicted_telescope = np.concatenate((window[-1:], predicted_telescope))
        plt.plot(np.arange(len(window) - 1, len(window) - 1 + len(predicted_telescope)), predicted_telescope,
                 label="predicted telescope")
    plt.legend()
    plt.show()

data = [training_data[i, :series_lengths[i]] for i in range(len(training_data))]

X, y, categories = build_sequence(data, categories, window_size=200, stride=200, telescope=18)

assert len(X) == len(y) == len(categories) == num_windows

# plot a random window and telescope
index = np.random.randint(0, len(X))
plot_window_telescope(X[index], y[index])

""" TESTED AND WORKING
# split the data into train and test
X_train, X_test, y_train, y_test, categories_train, categories_test, idx_train, idx_test = train_test_split(X, y,
                                                                                                            categories,
                                                                                                            range(
                                                                                                                len(X)),
                                                                                                            test_size=0.2,
                                                                                                            stratify=categories)
# assert the categories are the same as before splitting
for i in range(len(X_train)):
    assert categories_train[i] == categories[idx_train[i]]
"""
# split the data into train and test while preserving the categories distribution
X_train, X_test, y_train, y_test, categories_train, categories_test = train_test_split(X, y, categories,
                                                                                       test_size=0.2,
                                                                                       stratify=categories)

print("Number of training examples: ", len(X_train))
print("Number of test examples: ", len(X_test))
