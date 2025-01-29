import gdown
import os
import zipfile
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from scipy.spatial import KDTree
import numpy as np


def down_n_extract(id="1-1OSGlN2EOqyZuehBgpgI8FNOtK-caYf", directory="data"):
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

    def make_kdtree(self, model, savefile=None):
        data = np.array([[0.0] * len(self.dataset.classes) for _ in self.dataset])
        for i in range(len(self.dataset)):
            data[i] = model.get_ouput(self.dataset[i][0])

        tree = KDTree(data)
        self.tree = tree
        if savefile:
            np.save(savefile, data)

    def load_kdtree(self, datafile):
        data = np.load(datafile)
        self.tree = KDTree(data)

    def get_similar_images_indices(self, img, model, k=5):
        img = model.get_ouput(img)
        _, indices = self.tree.query(img.reshape(1, -1), k=k)
        return indices[0]

    def indices_to_images(self, indices):
        return [self.dataset[i][0].permute(1, 2, 0) for i in indices]
