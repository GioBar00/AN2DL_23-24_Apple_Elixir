def plot_image(img, lbl):
    plt.figure(1, 1, figsize=(96,96))
    plt.imshow(np.clip(img, 0, 1))
    plt.title(f'{lbl}')