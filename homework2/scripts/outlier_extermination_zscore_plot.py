import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

def remove_outliers_interpolate(series, threshold):
    # copy series
    series = series.copy()
    # Calculate zscore
    z_scores = zscore(series)
    # Identify outliers based on the threshold
    outliers = np.abs(z_scores) > threshold

    if np.sum(outliers) == 0:
        return False, series

    # Interpolate the outliers
    indices = np.arange(len(series))
    # Print outlier indices
    print("Outlier indices: ", np.where(outliers)[0])

    old_values = series[outliers]

    # Interpolate the outliers
    series[outliers] = np.interp(indices[outliers], indices[~outliers], series[~outliers])

    new_values = series[outliers]
    print("Old values: ", old_values)
    print("New values: ", new_values)

    return True, series

training_data = np.load("../dataset/training_data.npy", allow_pickle=True)
categories = np.load("../dataset/categories.npy", allow_pickle=True)
valid_periods = np.load("../dataset/valid_periods.npy", allow_pickle=True)

unpadded_data = []
for i in range(len(training_data)):
    unpadded_data.append(training_data[i, valid_periods[i, 0]:valid_periods[i, 1]])

prev_i = -1
for i in range(len(unpadded_data)):
    outliers_present, new_series = remove_outliers_interpolate(unpadded_data[i], 10)
    if outliers_present:
        # plot original series and series without outliers with dots
        plt.plot(unpadded_data[i], label="original")
        plt.plot(new_series, label="without outliers")
        plt.scatter(np.where(unpadded_data[i] != new_series)[0], new_series[np.where(unpadded_data[i] != new_series)[0]], c="r", label="outliers")
        # rescale series ?????????
        #new_series = (new_series - np.min(new_series)) / (np.max(new_series) - np.min(new_series))
        # plot rescaled series
        #plt.plot(new_series, label="rescaled")
        plt.legend()
        plt.show()
        unpadded_data[i] = new_series
        prev_i = i

