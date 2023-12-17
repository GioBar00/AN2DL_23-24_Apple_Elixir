import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

outlier_threshold = 10

def recursive_filter_time_series(series, i, indices):
    if len(series) == 1:
        return []
    # remove indices where the length of the series is i
    indices_to_remove = []

    # remove series at indices_to_remove without using numpy
    new_series = []
    new_indices = []
    for j in range(len(series)):
        if len(series[j]) > i:
            new_series.append(series[j])
            new_indices.append(indices[j])
        else:
            indices_to_remove.append(indices[j])

    series = new_series
    indices = new_indices

    if len(series) == 0:
        return indices_to_remove[1:]

    # group by element i of the series
    groups_series = {}
    groups_indices = {}
    for k, s in enumerate(series):
        if s[i] in groups_series.keys():
            groups_series[s[i]].append(s)
            groups_indices[s[i]].append(indices[k])
        else:
            groups_series[s[i]] = [s]
            groups_indices[s[i]] = [indices[k]]

    for key in groups_series.keys():
        indices_to_remove += recursive_filter_time_series(groups_series[key], i + 1, groups_indices[key])

    return indices_to_remove

def remove_outliers_interpolate(series, threshold):
    # Calculate zscore
    z_scores = zscore(series)
    # Identify outliers based on the threshold
    outliers = np.abs(z_scores) > threshold

    if np.sum(outliers) == 0:
        return False

    # Interpolate the outliers
    indices = np.arange(len(series))

    # Interpolate the outliers
    series[outliers] = np.interp(indices[outliers], indices[~outliers], series[~outliers])
    return True

training_data = np.load("../dataset/training_data.npy", allow_pickle=True)
old_categories = np.load("../dataset/categories.npy", allow_pickle=True)
old_valid_periods = np.load("../dataset/valid_periods.npy", allow_pickle=True)

old_unpadded_data = []
for i in range(len(training_data)):
    old_unpadded_data.append(training_data[i, old_valid_periods[i, 0]:old_valid_periods[i, 1]])

# remove duplicates
# indices_to_remove = filter_time_series(old_unpadded_data)
indices_to_remove = recursive_filter_time_series(old_unpadded_data, 0, np.arange(len(old_unpadded_data)))
unpadded_data = []
for i in range(len(old_unpadded_data)):
    if i not in indices_to_remove:
        unpadded_data.append(old_unpadded_data[i])
categories = np.delete(old_categories, indices_to_remove)
valid_periods = np.delete(old_valid_periods, indices_to_remove, axis=0)
print("Number of series contained in other series: ", len(indices_to_remove))
print("Indices of series contained in other series: ", indices_to_remove)

for i in range(len(unpadded_data)):
    res = remove_outliers_interpolate(unpadded_data[i], outlier_threshold)
    if res:
        print("Outliers removed for series ", i)

# get longest series length
max_length = 0
for series in unpadded_data:
    if len(series) > max_length:
        max_length = len(series)

# pad series right to max length with -1
padded_data = []
length = []
for series in unpadded_data:
    length.append(len(series))
    padded_data.append(np.pad(series, (0, max_length - len(series)), 'constant', constant_values=-1))

# change categories to integers
unique_categories = np.unique(categories)
for i in range(len(categories)):
    categories[i] = np.where(unique_categories == categories[i])[0][0]

# change categories to integer type
categories = categories.astype(int)


# assert old categories are the same as new categories for the same series
j = 0
for i in range(len(training_data)):
    if i not in indices_to_remove:
        original_cat = np.where(unique_categories == old_categories[i])[0][0]
        assert original_cat - categories[j] == 0
        j += 1


# save data
np.save("../dataset/training_data_clean.npy", padded_data)
np.save("../dataset/categories_clean.npy", categories)
np.save("../dataset/series_length_clean.npy", length)
