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

        self.linear1 = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )

        # Classification branch
        self.linear2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

        # Angle branch
        self.linear3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        
        self.linear4 = nn.Linear(32, 10)
        self.linear5 = nn.Linear(32, 1)

    def forward(self, X):
        h1 = self.linear1(X)
        
        h2 = self.linear2(h1)
        h3 = self.linear3(h1)
        
        h4 = self.linear4(h2)
        h5 = self.linear5(h3).view(-1)

        return h4, h5

class NN:
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
        X, y, a = torch.from_numpy(X).float(), torch.from_numpy(y_class), torch.from_numpy(y_angle)

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
        X = torch.from_numpy(X).float()

        y_pred, a_pred = self.model.forward(X)
        y_pred = torch.argmax(y_pred, dim=1)

        return y_pred.detach().numpy(), a_pred.detach().numpy()

    def fit(self, X, y_class, y_angle, step=1e-4):
        """Train the model according to the given training data.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Labels of objects. Each entry is in [0,...,C-1].
            y_angle (numpy ndarray, shape = (samples, )):
                Angles of the objects in degrees.
        """
        X, y, a = torch.from_numpy(X).float(), torch.from_numpy(y_class), torch.from_numpy(y_angle)
    
        dataset = TensorDataset(X, y, a)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=step, weight_decay=1e-4)

        for epoch in range(self.epochs):
            loss = None
            for X_batch, y_batch, a_batch in train_loader:
                loss = self.objective(X_batch.numpy(), y_batch.numpy(), a_batch.numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(loss.item())

    def get_params(self):
        """Get the model parameters.

        Returns:
            a list containing the following parameter values represented
            as numpy arrays (see handout for definitions of each parameter). 

            w1 (numpy ndarray, shape = (784, 256))
            b1 (numpy ndarray, shape = (256,))
            w2 (numpy ndarray, shape = (256, 64))
            b2 (numpy ndarray, shape = (64,))
            w3 (numpy ndarray, shape = (64, 32))
            b3 (numpy ndarray, shape = (32,))
            w4 (numpy ndarray, shape = (64, 32))
            b4 (numpy ndarray, shape = (32,))
            w5 (numpy ndarray, shape = (32, 10))
            b5 (numpy ndarray, shape = (10,))
            w6 (numpy ndarray, shape = (32, 1))
            b6 (numpy ndarray, shape = (1,))
        """
        return [param.T.detach().numpy() for param in self.model.parameters()]

    def set_params(self, params):
        """Set the model parameters.

        Arguments:
            params is a list containing the following parameter values represented
            as numpy arrays (see handout for definitions of each parameter).

            w1 (numpy ndarray, shape = (784, 256))
            b1 (numpy ndarray, shape = (256,))
            w2 (numpy ndarray, shape = (256, 64))
            b2 (numpy ndarray, shape = (64,))
            w3 (numpy ndarray, shape = (64, 32))
            b3 (numpy ndarray, shape = (32,))
            w4 (numpy ndarray, shape = (64, 32))
            b4 (numpy ndarray, shape = (32,))
            w5 (numpy ndarray, shape = (32, 10))
            b5 (numpy ndarray, shape = (10,))
            w6 (numpy ndarray, shape = (32, 1))
            b6 (numpy ndarray, shape = (1,))
        """
        new_state_dict = dict()
        for (key, _), param in zip(self.model.named_parameters(), params):
            new_state_dict[key] = nn.Parameter(torch.from_numpy(param.T))
        self.model.load_state_dict(new_state_dict)

def main():
    DATA_DIR = '../data'
    data=np.load(os.path.join(DATA_DIR, "mnist_rot_train.npz"))
    X_tr,y_tr,a_tr = data["X"],data["labels"],data["angles"]

    data=np.load(os.path.join(DATA_DIR, "mnist_rot_validation.npz"))
    X_val,y_val,a_val = data["X"],data["labels"],data["angles"]

    #Note: test class labels and angles are not provided
    #in the data set
#     data=np.load(os.path.join(DATA_DIR, "mnist_rot_test.npz"))
#     X_te,y_te,a_te = data["X"],data["labels"],data["angles"]
    
    model = NN(0.5, 20)
    model.fit(X_tr, y_tr, a_tr)

    y_cl, y_a = model.predict(X_tr)
    print(y_cl[:10], y_tr[:10])
    print('Train classification acc', (y_cl == y_tr).mean())
    
#     y_cl, y_a = model.predict(X_val)
#     print(y_cl[:10], y_tr[:10])
#     print('Val classification acc', (y_cl == y_val).mean())

if __name__ == '__main__':
    main()