import numpy as np
import matplotlib.pyplot as plt

def remove_outliers_interpolate(series, window_size, threshold):
    # copy series
    series = series.copy()
    # window size is the min between the length of the series and the window size
    window_size = min(len(series), window_size)
    # Calculate the moving average
    moving_avg = np.convolve(series, np.ones(window_size)/window_size, mode='same')

    # Calculate the measure of dispersion (standard deviation in this case)
    std_dev = np.std(series)

    # Identify outliers based on the threshold
    outliers = np.abs(series - moving_avg) > threshold * std_dev

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

for i in range(len(unpadded_data)):
    outliers_present, new_series = remove_outliers_interpolate(unpadded_data[i], 100, 10)
    if outliers_present:
        # plot original series and series without outliers with dots
        plt.plot(unpadded_data[i], label="original")
        plt.plot(new_series, label="without outliers")
        plt.scatter(np.where(unpadded_data[i] != new_series)[0], new_series[np.where(unpadded_data[i] != new_series)[0]], c="r", label="outliers")
        # rescale series ?????????
        new_series = (new_series - np.min(new_series)) / (np.max(new_series) - np.min(new_series))
        # plot rescaled series
        plt.plot(new_series, label="rescaled")
        plt.legend()
        plt.show()
        unpadded_data[i] = new_series

