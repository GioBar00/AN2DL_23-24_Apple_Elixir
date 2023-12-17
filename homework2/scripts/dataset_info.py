import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

def build_sequences(df, window=200, stride=200, telescope=100):
    # Sanity check to avoid runtime errors
    assert window % stride == 0
    dataset = []
    labels = []
    for i in range(len(df)):
        # Get the element
        element = df[i]
        # Get the length of the element
        length = len(element)
        # Get the number of sequences even partial ones
        num_sequences = length // stride + 1
        for j in range(num_sequences):
            # Get the start and end index of the sequence
            start_idx = j * stride
            end_idx = start_idx + window
            # Check if the sequence is shorter than the window
            if end_idx + telescope > length:
                sequence = element[start_idx: -telescope]
                label = element[-telescope:]
                dataset.append(sequence)
                labels.append(label)
                break
            else:
                # Get the sequence
                sequence = element[start_idx:end_idx]
                label = element[end_idx: end_idx + telescope]
                # Append the sequence to the dataset
                dataset.append(sequence)
                labels.append(label)

    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels

def plot_series(series, title):
    """
    Plot a series
    :param series: series to plot
    :param title: title of the plot
    :return: None
    """
    series.plot(figsize=(10, 6))
    plt.title(title)
    plt.show()

training_data = np.load("../dataset/training_data.npy", allow_pickle=True)
categories = np.load("../dataset/categories.npy", allow_pickle=True)
valid_periods = np.load("../dataset/valid_periods.npy", allow_pickle=True)

# print dataset properties
print("Number of training examples: ", len(training_data))
print("Number of categories: ", len(np.unique(categories)))

# average series length calculated by valid periods
print("Average series length: ", np.mean(valid_periods[:, 1] - valid_periods[:, 0]))
# series with length < 200
print("Number of series with length < 200: ", np.sum(valid_periods[:, 1] - valid_periods[:, 0] < 200))
print("Percentage of series with length < 200: ",
      np.sum(valid_periods[:, 1] - valid_periods[:, 0] < 200) / len(training_data))
print("*" * 50)
series_by_category = {}
print("Categories: ", np.unique(categories))
# for each category
for category in np.unique(categories):
    # number of series
    print("Number of series in category {}: {}".format(category, np.sum(categories == category)))
    # average series length
    print("Average series length in category {}: {}".format(category, np.mean(
        valid_periods[categories == category, 1] - valid_periods[categories == category, 0])))
    # series with length < 200
    print("Number of series with length < 200 in category {}: {}".format(category, np.sum(
        valid_periods[categories == category, 1] - valid_periods[categories == category, 0] < 200)))
    print("Percentage of series with length < 200 in category {}: {}".format(category, np.sum(
        valid_periods[categories == category, 1] - valid_periods[categories == category, 0] < 200) / np.sum(
        categories == category)))
    print("Number of series with length < 40 in category {}: {}".format(category, np.sum(
        valid_periods[categories == category, 1] - valid_periods[categories == category, 0] < 40)))
    print("Percentage of series with length < 40 in category {}: {}".format(category, np.sum(
        valid_periods[categories == category, 1] - valid_periods[categories == category, 0] < 40) / np.sum(
        categories == category)))
    # calculate average and std of series values
    padded_data = training_data[categories == category]
    periods = valid_periods[categories == category]
    data = []
    data_mean = []
    data_std = []
    data_len = []
    for i in range(len(padded_data)):
        data.append(padded_data[i, periods[i, 0]:periods[i, 1]])
        data_mean.append(np.mean(data[-1]))
        data_std.append(np.std(data[-1]))
        data_len.append(len(data[-1]))
    series_by_category[category] = data
    data_concat = np.concatenate(data)
    print("Average of series values in category {}: {}".format(category, np.mean(data_concat)))
    print("Std of series values in category {}: {}".format(category, np.std(data_concat)))
    print("Average of series mean in category {}: {}".format(category, np.mean(data_mean)))
    print("Std of series mean in category {}: {}".format(category, np.std(data_mean)))
    print("*" * 50)
    rand_idx = np.random.randint(0, len(padded_data))
    # plot one series
    #plot_series(pd.Series(padded_data[rand_idx, periods[rand_idx, 0]:periods[rand_idx, 1]]), "Series in category {}".format(category))

    # scale data ***(NOT USEFUL IT SEEMS)***
    # scaler = RobustScaler()
    # scaled_data = scaler.fit_transform(data_concat.reshape(-1, 1)).reshape(-1)
    # print("Average of scaled series values in category {}: {}".format(category, np.mean(scaled_data)))
    # print("Std of scaled series values in category {}: {}".format(category, np.std(scaled_data)))
    # print("*" * 50)
    # # plot one scaled series
    # initial_idx = np.cumsum(data_len)[rand_idx]
    # plot_series(pd.Series(scaled_data[initial_idx : initial_idx + data_len[rand_idx]]), "Scaled series in category {}".format(category))

    # X, y = build_sequences(data, window=200, stride=200, telescope=18)
    # plot_series(pd.Series(data[rand_idx]), "Series in category {}".format(category))
    # plot_series(pd.Series(X[rand_idx]), "Sequence in category {}".format(category))
    # plot_series(pd.Series(y[rand_idx]), "Label in category {}".format(category))

#plot how many series are for each length
# series_length = []
# for i in range(len(training_data)):
#     size = valid_periods[i, 1] - valid_periods[i, 0]
#     if size < 200:
#         series_length.append(size)
#
# plt.hist(series_length, bins=100)
# plt.show()

# import statsmodels.api as sm
#
# for category in np.unique(categories):
#     idexes = []
#     for i in range(len(series_by_category[category])):
#         acorr = sm.tsa.acf(series_by_category[category][i], nlags=200)
#         # last index where acorr is > 0.1
#         idx = np.where(abs(acorr) > 0.2)[0][-1]
#         idexes.append(idx)
#
#     indexes = np.array(idexes)
#     idxs_mean = np.mean(indexes, axis=0)
#     print("Category: {}".format(category))
#     print("Average autocorrelation: {}".format(idxs_mean))
#     # plot idxs
#     plt.hist(indexes, bins=100)
#     plt.show()

def plot_two_series(series1, series2, title):
    # plot 2 series in grid
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(series1)
    axes[0].set_title("First series length: {}".format(len(series1)))
    axes[1].plot(series2)
    axes[1].set_title("Second series length: {}".format(len(series2)))
    plt.suptitle(title)
    plt.show()

plot_idxs = [33257, 34474]
plot_two_series(training_data[plot_idxs[0], valid_periods[plot_idxs[0], 0]:valid_periods[plot_idxs[0], 1]],
                training_data[plot_idxs[1], valid_periods[plot_idxs[1], 0]:valid_periods[plot_idxs[1], 1]],
                "Equal series")
plot_idxs = [29300, 29317]
plot_two_series(training_data[plot_idxs[0], valid_periods[plot_idxs[0], 0]:valid_periods[plot_idxs[0], 1]],
                training_data[plot_idxs[1], valid_periods[plot_idxs[1], 0]:valid_periods[plot_idxs[1], 1]],
                "First series contained in second series")

exit(0)
# for each category plot a 4 random series in a grid
fig, axes = plt.subplots(len(np.unique(categories)), 4, figsize=(20, 20))
for i, category in enumerate(np.unique(categories)):
    for j in range(4):
        rand_idx = np.random.randint(0, len(series_by_category[category]))
        axes[i, j].plot(series_by_category[category][rand_idx])
        axes[i, j].set_title("Category: {}".format(category))
plt.show()

# build sequences