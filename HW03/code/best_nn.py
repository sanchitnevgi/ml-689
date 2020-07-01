import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import gzip
import pickle
import os

np.random.seed(1)
torch.manual_seed(1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 7 * 7, 64),
            nn.ReLU(inplace=True),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10)
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, X):
        h1 = self.conv1(X)
        h2 = self.conv2(h1)
        h2 = torch.flatten(h2, start_dim=1)

        h3 = self.fc1(h2)
        
        h4 = self.fc2(h3)
        h5 = self.fc3(h3)

        return h4, h5.view(-1)

    def predict(self, X):
        y_pred, a_pred = self.forward(X)
        y_pred = torch.argmax(y_pred, dim=1)

        return y_pred, a_pred.view(-1)

class BestNN:
    """A network architecture for simultaneous classification 
    and angle regression of objects in images.

    Arguments:
        alpha: trade-off parameter for the composite objective function.
        epochs: number of epochs for training
    """
    def __init__(self, alpha=.5, epochs=5):
        self.alpha = alpha
        self.epochs = epochs
        
        self.model = Net()


    def objective(self, X, y_class, y_angle):
        """Objective function.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Labels of objects. Each entry is in [0,...,C-1].
            y_angle (numpy ndarray, shape = (samples, )):
                Angles of the objects in degrees.

        Returns:
            Composite objective function value.
        """
        X, y, a = torch.from_numpy(X).float().view(-1, 1, 28, 28), torch.from_numpy(y_class), torch.from_numpy(y_angle)

        y_pred, a_pred = self.model.forward(X)

        ce_loss = F.cross_entropy(y_pred, y, reduction='sum')
        a_loss = 0.5 * (1 - torch.cos(0.01745 * (a - a_pred))).sum()

        loss = (self.alpha * ce_loss) + (1 - self.alpha) * a_loss

        return loss.item()
        

    def predict(self, X):
        """Predict class labels and object angles for samples in X.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Input matrix where each row is a feature vector.

        Returns:
            y_class (numpy ndarray, shape = (samples,)):
                predicted labels. Each entry is in [0,...,C-1].
            y_angle (numpy ndarray, shape = (samples, )):
                The predicted angles of the imput objects.
        """
        X = torch.from_numpy(X).float().view(-1, 1, 28, 28)

        y_pred, a_pred = self.model.forward(X)
        y_pred = torch.argmax(y_pred, dim=1)

        return y_pred, a_pred

    def fit(self, X, y_class, y_angle ,step=1e-4):
        """Train the model according to the given training data.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Labels of objects. Each entry is in [0,...,C-1].
            y_angle (numpy ndarray, shape = (samples, )):
                Angles of the objects in degrees.
        """
        X, y, a = torch.from_numpy(X).float().view(-1, 1, 28, 28), torch.from_numpy(y_class), torch.from_numpy(y_angle)

        alpha = self.alpha
        optimizer = torch.optim.Adam(self.model.parameters(), lr=step, weight_decay=1e-4)
        dataset = TensorDataset(X, y, a)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(self.epochs):
            loss = None
            for i, (X_batch, y_batch, a_batch) in enumerate(train_loader):

                y_pred, a_pred = self.model.forward(X_batch)

                ce_loss = F.cross_entropy(y_pred, y_batch, reduction='sum')
                a_loss = 0.5 * (1 - torch.cos(0.01745 * (a_batch - a_pred))).sum()
                loss = (alpha * ce_loss) + (1 - alpha) * a_loss

                if i % 20 == 0:
                    print(epoch, i, loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        


def main():
    
    DATA_DIR = '../data'
    data=np.load(os.path.join(DATA_DIR, "mnist_rot_train.npz"))
    X_tr,y_tr,a_tr = data["X"],data["labels"],data["angles"]

    data=np.load(os.path.join(DATA_DIR, "mnist_rot_validation.npz"))
    X_val,y_val,a_val = data["X"],data["labels"],data["angles"]

    #Note: test class labels and angles are not provided
    #in the data set
    data=np.load(os.path.join(DATA_DIR, "mnist_rot_test.npz"))
    X_te,y_te,a_te = data["X"],data["labels"],data["angles"]
    
    nn = BestNN(0.2, 20)
    nn.fit(X_tr, y_tr, a_tr)
    
if __name__ == '__main__':
    main()