import gdown
import os
import zipfile
import matplotlib.pyplot as plt
import sqlite3
from torchvision import datasets, transforms
from scipy.spatial import KDTree
import numpy as np


def down_n_extract(id="1-1OSGlN2EOqyZuehBgpgI8FNOtK-caYf", directory="data"):
    """
    Downloads a zip file from Google Drive and extracts it to a given directory.

    Parameters
    ----------
    id : str, optional
        The id of the zip file in Google Drive. Defaults to "1-1OSGlN2EOqyZuehBgpgI8FNOtK-caYf".
    directory : str, optional
        The path to the directory where the zip file should be extracted. Defaults to "data".

    Returns
    -------
    None
    """

    zip_path = f"{directory}.zip"
    if not os.path.exists(directory):
        if not os.path.exists(zip_path):
            gdown.download(id=id, output=zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(directory)
    else:
        print("Directory already exists")


class Dataset:
    def __init__(self, data_path):
        transform = transforms.Compose(
            [
                transforms.Resize((256, 180)),
                transforms.ToTensor(),
            ]
        )
        transform_train = transforms.Compose(
            [
                transforms.Resize((256, 180)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.dataset = datasets.ImageFolder(root=data_path, transform=transform)
        self.data_train = datasets.ImageFolder(
            root=data_path, transform=transform_train
        )

    def show_images(self, indices: list):
        """
        Display images from the dataset with their labels.

        Args:
            indices (list): A list of indices to select images from the dataset.

        The function plots the images in a grid with their labels.
        """
        fig, axes = plt.subplots(
            int((len(indices) + 3) / 4),
            4,
            figsize=(12, int((len(indices) + 3) / 4) * 4),
        )
        axes = axes.flatten()
        for i, ax in enumerate(axes[: len(indices)]):
            img, label = self.dataset[indices[i]]
            ax.imshow(img.permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
            ax.set_title(self.dataset.classes[label])
            ax.axis("off")
        for ax in axes[len(indices) :]:
            ax.axis("off")
        fig.tight_layout()
        plt.show()

    def show_images_with_result(self, indices: list, model):
        """
        Display images with their true labels and model's predicted labels.

        Args:
            indices (list): A list of indices to select images from the dataset.
            model: A trained model used to predict the class of each image. Need predict() method

        The function plots the images in a grid with their true labels and the model's predicted labels.
        """

        fig, axes = plt.subplots(
            int((len(indices) + 3) / 4),
            4,
            figsize=(12, int((len(indices) + 3) / 4) * 4),
        )
        axes = axes.flatten()
        for i, ax in enumerate(axes[: len(indices)]):
            img, label = self.dataset[indices[i]]
            predict = model.predict(img)
            ax.imshow(img.permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
            ax.set_title(
                f"TG: {self.dataset.classes[label]} \n Model: {self.dataset.classes[predict]}"
            )
            ax.axis("off")
        for ax in axes[len(indices) :]:
            ax.axis("off")
        fig.tight_layout()
        plt.show()

    def make_kdtree(self, model, table="KDTree_Film", savefile=None):
        data = np.array([[0.0] * len(self.dataset.classes) for _ in self.dataset])
        for i in range(len(self.dataset)):
            data[i] = model.get_ouput(self.dataset[i][0])

        data_tree = KDTree(data)
        self.tree = data_tree
        if savefile:
            self.save_to_db(savefile, data, table)

    def save_to_db(self, database, data=None, table="KDTree_Film"):
        conn = sqlite3.connect(database)
        cursor = conn.cursor()

        keys = (
            "(id INTEGER PRIMARY KEY "
            + ", ".join(
                [
                    f"{self.dataset.classes[i]} REAL"
                    for i in range(len(self.dataset.classes))
                ]
            )
            + ")"
        )

        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} {keys}")

        for i in range(len(data)):
            values = tuple(data[i])
            cursor.execute(
                f"INSERT INTO KDTree_Film VALUES ({i}, {', '.join(['?' for _ in range(len(self.dataset.classes))])})",
                values,
            )

        conn.commit()
        conn.close()

    def load_kdtree(self, database: str, table="KDTree_Film"):
        conn = sqlite3.connect(database)
        cursor = conn.cursor()

        query = f"SELECT {', '.join(self.dataset.classes)} FROM {table}"  # Modify query as needed
        cursor.execute(query)
        rows = cursor.fetchall()
        data_from_db = np.array(rows)
        data_tree = KDTree(data_from_db)
        self.tree = data_tree

    def get_similar_images_indices(self, img, model, k=5):
        img = model.get_ouput(img)
        _, indices = self.tree.query(img.reshape(1, -1), k=k)
        return indices[0]

    def indices_to_images(self, indices):
        return [self.dataset[i][0].permute(1, 2, 0) for i in indices]
