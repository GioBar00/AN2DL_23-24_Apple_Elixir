import numpy as np

# CLEAN BALANCED SPLITTED
X_tr_val, y_tr_val, X_test, y_test = np.load('public_data_clean_balanced_splitted.npz', allow_pickle=True).values()

print('X_tr_val: ', X_tr_val.shape)
print('y_tr_val: ', y_tr_val.shape)
print('X_test: ', X_test.shape)
print('y_test: ', y_test.shape)
print('-' * 20 + 'DISTRIBUTION' + '-' * 20)
print('Healthy images train: ', len(X_tr_val[y_tr_val == 0]))
print('Unhealthy images train: ', len(X_tr_val[y_tr_val == 1]))
print('Healthy images test: ', len(X_test[y_test == 0]))
print('Unhealthy images test: ', len(X_test[y_test == 1]))

# CLEAN
# images, labels = np.load('public_data_clean.npz', allow_pickle=True).values()
# print('Images: ', len(images))
# print('Labels: ', len(labels))
# print('Healthy images: ', len(images[labels == 'healthy']))
# print('Unhealthy images: ', len(images[labels == 'unhealthy']))