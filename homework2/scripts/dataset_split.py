import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

training_data = np.load("../dataset/training_data.npy", allow_pickle=True)
categories = np.load("../dataset/categories.npy", allow_pickle=True)
valid_periods = np.load("../dataset/valid_periods.npy", allow_pickle=True)


def build_sequence(data, window_size=200, stride=200, telescope=18):
    actual_window = window_size + telescope

    # two dimensional array (number of series, window size)
    X = []
    # two dimensional array (number of series, telescope
    y = []

    for i, element in enumerate(data):
        length = len(element)
        # number of windows
        n_windows = int(np.ceil((length - actual_window) / stride)) + 1
        # reevaluate the stride
        if n_windows > 1:
            stride = int((length - actual_window) / (n_windows - 1)) + 1
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
            # update the start and end index
            start_idx -= stride
            end_idx -= stride

    return X, y

def plot_window_telescope(window, telescope, predicted_telescope=None):
    # remove padding if present
    window = window[window != -1.]
    # connect window and telescope
    window = np.concatenate((window, telescope[:1]))
    # plot window and telescope
    plt.plot(np.arange(len(window)), window, label="window")
    plt.plot(np.arange(len(window) - 1, len(window) - 1 + len(telescope)), telescope, label="telescope")
    if predicted_telescope is not None:
        plt.plot(np.arange(len(window), len(window) + len(predicted_telescope)), predicted_telescope, label="predicted telescope")
    plt.legend()
    plt.show()


data = [training_data[i, valid_periods[i, 0]:valid_periods[i, 1]] for i in range(len(training_data))]

X, y = build_sequence(data, window_size=200, stride=200, telescope=18)


# plot a random window and telescope
index = np.random.randint(0, len(X))
plot_window_telescope(X[index], y[index])

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Number of training examples: ", len(X_train))
print("Number of test examples: ", len(X_test))


