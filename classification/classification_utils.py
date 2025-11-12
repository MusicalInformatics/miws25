from typing import Optional, Union, List
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


COMPOSER_CLASSES = {
    "Claude Debussy": 0,
    "Franz Liszt": 1,
    "Franz Schubert": 2,
    "Johann Sebastian Bach": 3,
    "Ludwig van Beethoven": 4,
    "Maurice Ravel": 5,
    "Robert Schumann": 6,
    "Sergei Rachmaninoff": 7,
    "Wolfgang Amadeus Mozart": 8,
}

CLASSES_COMPOSER = {
    0: "Claude Debussy",
    1: "Franz Liszt",
    2: "Franz Schubert",
    3: "Johann Sebastian Bach",
    4: "Ludwig van Beethoven",
    5: "Maurice Ravel",
    6: "Robert Schumann",
    7: "Sergei Rachmaninoff",
    8: "Wolfgang Amadeus Mozart",
}


def encode_composer(composers: Union[np.ndarray, str, pd.DataFrame]) -> np.ndarray:

    if isinstance(composers, str):

        return np.array(
            [COMPOSER_CLASSES[composers]],
            dtype=int,
        )

    else:

        composer_indices = np.array(
            [COMPOSER_CLASSES[cmp] for cmp in composers],
            dtype=int,
        )

        return composer_indices


def decode_composer(composer_index: Union[np.ndarray, int]) -> np.ndarray:

    if isinstance(composer_index, np.ndarray):
        return np.array(
            [CLASSES_COMPOSER[idx] for idx in composer_index],
            dtype="U256",
        )

    else:
        return np.array(
            [CLASSES_COMPOSER[composer_index]],
            dtype="U256",
        )


def segment_array(array: np.ndarray, window_length: int) -> np.ndarray:
    """
    Segments a 2D numpy array into non-overlapping windows of a specified length.
    Pads the end of the array with zeros along the first axis if necessary.

    Parameters
    -----------
    array : np.ndaarray
        2D array of shape (length, 88).

    window_length: int
        Length of each segment window.

    Returns
    -------
    segmented_array: np.ndarray
        Segmented 3D array of shape (n_windows, window_length, 88).
    """
    length = array.shape[0]
    n_windows = np.ceil(length / window_length).astype(int)
    padded_length = n_windows * window_length

    # Pad the array with zeros along the first axis
    padded_array = np.zeros((padded_length, array.shape[1]))
    padded_array[:length, :] = array

    # Reshape into non-overlapping windows
    segmented_array = padded_array.reshape(n_windows, window_length, array.shape[1])

    return segmented_array


def predict_piece(
    pianoroll: np.ndarray,
    segment_length: int,
    cnn: nn.Module,
    device: Optional[torch.device] = None,
) -> int:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    cnn.to(device)

    cnn.eval()

    segmented_pr = segment_array(
        array=pianoroll,
        window_length=segment_length,
    )

    segmented_pr_tensor = (
        torch.tensor(
            segmented_pr,
            dtype=torch.float32,
        )
        .unsqueeze(1)
        .to(device)
    )

    output = cnn(segmented_pr_tensor)

    _, preds = torch.max(output.data, 1)

    preds = preds.cpu().numpy()

    # Agregate using the most common class
    unique_labels, counts = np.unique(
        preds,
        return_counts=True,
    )
    mode_index = counts.argmax()
    mode = unique_labels[mode_index]
    return mode


def plot_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot a confusion matrix using matplotlib.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted labels (1D array).
    targets : np.ndarray
        True labels (1D array).
    class_names : list of str, optional
        List of class names corresponding to label indices.
        If None, class indices are used as labels.
    normalize : bool, optional
        If True, normalize the confusion matrix by true label counts.
    title : str, optional
        Title of the confusion matrix plot.

    Returns
    -------
    None
        Displays the confusion matrix plot.
    """
    # Compute unique classes and create a class index mapping
    classes = np.unique(np.concatenate((targets, predictions)))
    num_classes = len(classes)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    # Initialize the confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)

    # Populate the confusion matrix
    for t, p in zip(targets, predictions):
        i = class_to_index[t]
        j = class_to_index[p]
        cm[i, j] += 1

    # Normalize the confusion matrix if required
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)  # Replace NaNs with zeros

    # Use class names if provided
    if class_names is None:
        class_names = [str(cls) for cls in classes]
    else:
        if len(class_names) != num_classes:
            raise ValueError("Length of class_names must match number of classes")

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.Blues
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )

    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with the numeric value
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.show()
