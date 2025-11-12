#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline submission for the composer identification estimation challenge 
for Musical Informatics WS25
"""
from typing import Union
import pandas as pd
import numpy as np
import partitura as pt
import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from partitura.utils.music import compute_pianoroll

from sklearn.model_selection import train_test_split

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="partitura.*",
)

warnings.filterwarnings("ignore", module="sklearn")

from classification_utils import (
    encode_composer,
    decode_composer,
    segment_array,
    predict_piece,
    COMPOSER_CLASSES,
)


from sklearn.metrics import (
    accuracy_score,
    f1_score,
)



def compute_score_features(
    score: Union[pt.score.Score, pt.score.Part, np.ndarray]
) -> np.ndarray:

    pr = (
        compute_pianoroll(
            score,
            piano_range=True,
            time_unit="beat",
            time_div=time_div,
        )
        .toarray()
        .T
    )

    return pr


def load_composer_identification_dataset(datadir: str) -> np.ndarray:

    train_labels_fn = os.path.join(
        os.path.dirname(__file__),
        "composer_classification_training.csv",
    )

    train_data_ = pd.read_csv(train_labels_fn, delimiter=",")

    train_data = {}
    for bn, composer in zip(
        train_data_["Score"], encode_composer(train_data_["Composer"])
    ):
        fn = os.path.join(os.path.abspath(datadir), bn)
        if not os.path.exists(fn):
            raise ValueError(f"{fn} not found!")
        train_data[bn] = (composer, fn)

    # Test files without labels
    test_labels_fn = "./composer_classification_test_gt.csv"

    if not os.path.exists(test_labels_fn):
        test_labels_fn = "./composer_classification_test_no_labels.csv"

        test_data_ = pd.read_csv(test_labels_fn, delimiter=",")
        test_data = {}
        for bn in test_data_["Score"]:

            fn = os.path.join(os.path.abspath(datadir), bn)

            if not os.path.exists(fn):
                raise ValueError(f"{fn} not found!")
            test_data[bn] = (np.nan, fn)

    else:

        test_data_ = pd.read_csv(test_labels_fn, delimiter=",")

        test_data = {}
        for bn, composer in zip(
            test_data_["Score"], encode_composer(test_data_["Composer"])
        ):

            fn = os.path.join(os.path.abspath(datadir), bn)

            if not os.path.exists(fn):
                raise ValueError(f"{fn} not found!")
            test_data[bn] = (composer, fn)

    return train_data, test_data


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Difficulty estimation")

    parser.add_argument(
        "--datadir",
        "-i",
        help="path to the input files",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--outfn",
        "-o",
        help="Output file with results",
        type=str,
        default="composer_identification_results_cnn.csv",
    )

    args = parser.parse_args()

    if args.datadir is None:
        raise ValueError("No data directory given")

    train_data, test_data = load_composer_identification_dataset(args.datadir)

    # You should play with these hyper parameters!
    time_div = 10
    pr_segment_length = 16 * 10

    train_scores = []
    train_features = []
    train_labels = []
    for i, (score_name, (composer, fn)) in enumerate(train_data.items()):

        print(f"Processing training score {i + 1}/{len(train_data)}: {score_name}")
        score = pt.load_musicxml(fn)
        score[0].use_musical_beat()

        features = compute_score_features(score)

        train_scores.append(train_scores)
        train_features.append(features)
        train_labels.append(composer)

    train_features = np.array(train_features, dtype=object)
    train_labels = np.array(train_labels)

    val_size = 0.2
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features,
        train_labels,
        test_size=val_size,
        stratify=train_labels,
        random_state=42,
    )

    # length of the segment
    X_train = []
    Y_train = []

    for pr, ci in zip(train_features, train_labels):

        segmented_pr = segment_array(
            array=pr,
            window_length=pr_segment_length,
        )
        targets = np.ones(len(segmented_pr)) * ci

        X_train.append(segmented_pr)
        Y_train.append(targets)

    # Concatenate the arrays
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)

    # length of the segment
    X_val = []
    Y_val = []

    for pr, ci in zip(val_features, val_labels):

        segmented_pr = segment_array(
            array=pr,
            window_length=pr_segment_length,
        )
        targets = np.ones(len(segmented_pr)) * ci

        X_val.append(segmented_pr)
        Y_val.append(targets)

    # Concatenate the arrays
    X_val = np.concatenate(X_val)
    Y_val = np.concatenate(Y_val)

    print("Training classifier...")

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(
        1
    )  # Add channel dimension
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(
        1
    )  # Add channel dimension
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)

    n_classes = len(COMPOSER_CLASSES)

    # Hyperparameters
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 10

    # Create DataLoader for training and valing sets
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    class CNN(nn.Module):
        def __init__(
            self,
            n_classes=n_classes,
            input_height=X_train.shape[1],
            input_width=X_train.shape[2],
        ):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(
                1, 32, kernel_size=3, stride=1, padding=1
            )  # Output: (batch_size, 32, input_height, input_width)
            self.conv2 = nn.Conv2d(
                32, 64, kernel_size=3, stride=1, padding=1
            )  # Output: (batch_size, 64, input_height, input_width)
            self.pool = nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0
            )  # Halves the spatial dimensions
            self.relu = nn.ReLU()

            # Compute the size after conv and pooling layers
            self.flattened_size = self._get_flattened_size(input_height, input_width)

            # Now, initialize the fully connected layers with the correct input size
            self.fc1 = nn.Linear(self.flattened_size, 128)
            self.fc2 = nn.Linear(128, n_classes)

        def _get_flattened_size(self, height, width):
            # Pass a dummy input through conv and pool layers to get the output size
            with torch.no_grad():
                x = torch.zeros(1, 1, height, width)
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                flattened_size = x.numel()
            return flattened_size

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))  # After first conv and pool
            x = self.pool(self.relu(self.conv2(x)))  # After second conv and pool
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Instantiate the model, define the loss function and the optimizer
    model = CNN(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0  # Initialize best validation accuracy
    validate_freq = 1  # Set validation frequency (default is 1)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
        )
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": running_loss / (i + 1)})

        # Perform validation every 'validate_freq' epochs
        if (epoch + 1) % validate_freq == 0:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_loss /= len(val_loader)
            val_acc = 100 * correct / total
            print(
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%"
            )

            # Save checkpoint if validation accuracy improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_model_checkpoint.pth")
                print("Model checkpoint saved.")

    # Load the state dictionary from the saved file
    state_dict = torch.load(
        "best_model_checkpoint.pth",
        map_location=device,
    )
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # We load the scores in the test set.
    test_scores = []
    test_features = []
    test_labels = []
    score_names = []
    Y_pred = []
    for i, (score_name, (composer, fn)) in enumerate(test_data.items()):

        print(f"Processing test score {i + 1}/{len(test_data)}: {score_name}")
        score = pt.load_musicxml(fn)
        score[0].use_musical_beat()

        features = compute_score_features(score)

        pred = predict_piece(
            pianoroll=features,
            segment_length=pr_segment_length,
            cnn=model,
            device=device,
        )

        Y_pred.append(pred)

        test_scores.append(test_scores)
        test_features.append(features)
        test_labels.append(composer)
        score_names.append(score_name)

    Y_test = np.array(test_labels)
    Y_pred = np.array(Y_pred)
    score_names_test = np.array(score_names)

    # Run evaluation on the test labels if available
    if not any(np.isnan(Y_test)):
        print("#" * 55)
        print("Composer Identification Results on Test Set\n")
        acc = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred, average="macro")
        print(f"    Accuracy (test set): {acc:.2f}")
        print(f"    Macro F1-score (test set): {f1:.2f}")
        print("#" * 55)

    # This part will only save results for the test set!
    with open(args.outfn, "w") as f:

        f.write("file,difficulty\n")

        for basename, pred_dif in zip(score_names_test, decode_composer(Y_pred)):
            f.write(f"{basename},{pred_dif}\n")
