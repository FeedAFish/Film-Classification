from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.dataloader import Dataset


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, 2)
        self.cv2 = nn.Conv2d(32, 16, 2)
        self.cv3 = nn.Conv2d(16, 8, 2)
        self.flatten = nn.Flatten()
        conv_output_size = 253 * 177 * 8
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.cv1(x))
        x = F.relu(self.cv2(x))
        x = F.relu(self.cv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(
        self,
        data,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss(),
        save_file=None,
        epochs=10,
    ):
        if not data:
            return
        train_loader = torch.utils.data.DataLoader(
            data.data_train, batch_size=64, shuffle=True
        )
        opti = optimizer(self.parameters(), 1e-3)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                opti.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                opti.step()

                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(
                        "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000)
                    )
                    running_loss = 0.0

        if save_file:
            torch.save(self, save_file)
        print("Finished Training")
