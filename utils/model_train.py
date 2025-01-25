from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleNN(nn.Module):
    def __init__(self, output_channel=10):
        super(SimpleNN, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, 2)
        self.cv2 = nn.Conv2d(32, 16, 2)
        self.cv3 = nn.Conv2d(16, 8, 2)
        self.flatten = nn.Flatten()
        conv_output_size = 253 * 177 * 8
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_channel)
        self.classes = None

    def forward(self, x):
        x = F.relu(self.cv1(x))
        x = F.relu(self.cv2(x))
        x = F.relu(self.cv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_one_epoch(
        self,
        train_loader,
        optimizer,
        criterion,
    ):
        """
        Train model for one epoch.

        Args:
            train_loader: A DataLoader that loads in batches the data to train the model.
            optimizer: The optimizer to use to train the model.
            criterion: The loss function to use to calculate the loss.

        Returns:
            A tuple containing the average loss and accuracy of the model in this epoch.
        """
        train_loss = 0
        train_correct = 0
        counter = 0
        for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            counter += 1
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = self(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            _, pred = torch.max(outputs.data, 1)
            train_correct += (pred == labels).sum().item()

            loss.backward()
            optimizer.step()

        return train_loss / counter, 100.0 * train_correct / len(train_loader.dataset)

    def train_epochs(
        self,
        data,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss(),
        learning_rate=0.001,
        epochs: int = 10,
        save_file: str = None,
    ):
        if not self.classes:
            self.classes = data.data_train.classes
        if not set(data.data_train.classes).issubset(self.classes):
            print("New classes not recognised in new dataset")
            return
        train_loader = DataLoader(data.data_train, batch_size=64, shuffle=True)
        self.train()

        optimizer = optimizer(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss, train_acc = self.train_one_epoch(
                train_loader, optimizer, criterion
            )
            print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.2f}%")

        if save_file is not None:
            torch.save(
                {
                    "model_state_dict": self.state_dict(),
                    "classes": self.classes,  # Save the classes attribute
                },
                save_file,
            )

    @classmethod
    def load_model(cls, path: str):
        """Load model with its configuration"""
        checkpoint = torch.load(path, weights_only=False)
        model = cls()
        model.classes = checkpoint["classes"]
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def transform_image(self, img):
        transform = transforms.Compose(
            [
                transforms.Resize((256, 180)),
                transforms.ToTensor(),
            ]
        )
        return transform(img).unsqueeze(0)

    def predict(self, img):
        """Predict class of an image.

        Args:
            img: A PIL image.

        Returns:
            An integer representing the predicted class of the image."""
        with torch.no_grad():
            return torch.argmax(self(self.transform_image(img)))

    def predict_tensor(self, img):
        with torch.no_grad():
            return torch.argmax(self(img.unsqueeze(0)))

    def get_ouput(self, img):
        with torch.no_grad():
            return self(self.transform_image(img))
