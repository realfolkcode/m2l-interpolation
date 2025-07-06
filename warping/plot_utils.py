import librosa
import matplotlib.pyplot as plt


def plot_dtw(X,
             Y,
             P,
             X_chroma,
             Y_chroma,
             hop_length):
    """Plots DTW path between two embeddings."""
    N = X_chroma.shape[1]
    M = Y_chroma.shape[1]

    plt.figure(figsize=(8, 3))
    ax_X = plt.axes([0, 0.60, 1, 0.40])
    librosa.display.specshow(X_chroma, ax=ax_X, x_axis='frames', y_axis='chroma', cmap='gray_r', hop_length=hop_length)
    ax_X.set_ylabel('Sequence X')
    ax_X.set_xlabel('Time (frames)')
    ax_X.xaxis.tick_top()
    ax_X.xaxis.set_label_position('top') 

    ax_Y = plt.axes([0, 0, 1, 0.40])
    librosa.display.specshow(Y_chroma, ax=ax_Y, x_axis='frames', y_axis='chroma', cmap='gray_r', hop_length=hop_length)
    ax_Y.set_ylabel('Sequence Y')
    ax_Y.set_xlabel('Time (frames)')

    step = 5
    y_min_X, y_max_X = ax_X.get_ylim()
    y_min_Y, y_max_Y = ax_Y.get_ylim()

    X_coef = N / X.shape[1]
    Y_coef = M / Y.shape[1]
    for t in P[0:-1:step, :]: 
        ax_X.vlines(t[0] * X_coef, y_min_X, y_max_X, color='r')
        ax_Y.vlines(t[1] * Y_coef, y_min_Y, y_max_Y, color='r')

    ax = plt.axes([0, 0.40, 1, 0.20])
    for p in P[0:-1:step, :]: 
        ax.plot((p[0]/N * X_coef, p[1]/M * Y_coef), (1, -1), color='r')
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([]);
