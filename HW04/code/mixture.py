import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.special import softmax
import os

np.random.seed(1)
torch.manual_seed(1)

class mixture_model:
    """A Laplace mixture model trained using marginal likelihood maximization

    Arguments:
        K: number of mixture components
        
    """
    def __init__(self, K=5):
        self.K = K
        
    def get_params(self):
        """Get the model parameters.

        Returns:
            a list containing the following parameter values: 

            mu (numpy ndarray, shape = (D, K))
            b (numpy ndarray, shape = (D,K))
            pi (numpy ndarray, shape = (K,))

        """
        return [self.mu, np.exp(self.b), np.exp(self.pi) / 100]

    def set_params(self, mu, b, pi):
        """Set the model parameters.

        Arguments:
            mu (numpy ndarray, shape = (D, K))
            b (numpy ndarray, shape = (D,K))
            pi (numpy ndarray, shape = (K,))
        """
        self.mu = mu
        self.b = np.log(b)
        self.pi = np.log(pi) + np.log(100)

    def marginal_likelihood(self, X):
        """log marginal likelihood function.
           Computed using the current values of the parameters
           as set via fit or set_params.

        Arguments:
            X (numpy ndarray, shape = (N, D)):
                Input matrix where each row is a feature vector.
                Missing data is indicated by np.nan's

        Returns:
            Marginal likelihood of observed data
        """
        mu, b, pi = self.mu, self.b, self.pi
        N, D = X.shape

        pi_norm = softmax(pi)
        assert pi_norm.shape == (self.K,)

        mgl_likelihood = np.zeros((N, self.K))

        for k in range(self.K):
            mu_d, b_d = mu[:, k], np.exp(b[:, k])
            assert mu_d.shape == b_d.shape == (D,)

            p_xz = 1 / (2 * b_d) * np.exp(-np.abs(X - mu_d) / b_d)
            
            # Handle missing data
            p_nan = np.isnan(p_xz)
            p_xz[p_nan] = 1.
            
            assert p_xz.shape == (N, D)

            p_xz_z = p_xz.prod(axis=1) * pi_norm[k]
            assert p_xz_z.shape == (N,)

            mgl_likelihood[:,k] = p_xz_z
        
        mgl_likelihood = np.log(mgl_likelihood.sum(axis=1)).sum()

        return mgl_likelihood

    def predict_proba(self, X):
        """Predict the probability over clusters P(Z=z|X=x) for each example in X.
           Use the currently stored parameter values.

        Arguments:
            X (numpy ndarray, shape = (N,D)):
                Input matrix where each row is a feature vector.

        Returns:
            PZ (numpy ndarray, shape = (N,K)):
                Probability distribution over classes for each
                data case in the data set.
        """
        N, D = X.shape
        mu, pi, b = self.mu, self.pi, self.b

        # Normalized pi values. Shape (K,)
        pi_norm = softmax(pi)

        pz = np.zeros((N, self.K))

        for k in range(self.K):
            mu_d, b_d = mu[:, k], np.exp(b[:, k])
            assert mu_d.shape == b_d.shape == (D,)

            p_xz = 1 / (2 * b_d) * np.exp(-np.abs(X - mu_d) / b_d)

            p_nan = np.isnan(p_xz)
            p_xz[p_nan] = 1.

            assert p_xz.shape == (N, D)

            p_xz_z = p_xz.prod(axis=1) * pi_norm[k]
            assert p_xz_z.shape == (N,)

            pz[:, k] = p_xz_z

        pz = pz / pz.sum(axis=1, keepdims=True)

        return pz
        

    def impute(self, X):
        """Mean imputation of missing values in the input data matrix X.
           Impute based on the currently stored parameter values.    

        Arguments:
            X (numpy ndarray, shape = (N, D)):
                Input matrix where each row is a feature vector.
                Missing data is indicated by np.nan's
        
        Returns:
            XI (numpy ndarray, shape = (N, D)):
                The input data matrix where the missing values on
                each row (indicated by np.nans) have been imputed 
                using their conditional means given the observed
                values on each row.
        """        
        mu = self.mu

        pz = self.predict_proba(X)

        XI = np.nan_to_num(X)

        X_nan = np.isnan(X)
        
        pz_mean = pz @ mu.T

        XI = XI + (X_nan * pz_mean)

        return XI

    def fit(self, X, mu_init=None, b_init=None, pi_init=None, step=0.1, epochs=5):
        """Train the model according to the given training data
           by directly maximizing the marginal likelihood of
           the observed data. If initial parameters are specified, use those
           to initialize the model. Otherwise, use a random initialization.

        Arguments:
            X (numpy ndarray, shape = (N, D)):
                Input matrix where each row is a feature vector.
                Missing data is indicated by np.nan's
            mu_init (None or numpy ndarray, shape = (D, K)):
                Array of Laplace density mean paramaeters for each mixture component
                to use for initialization
            b_init (None or numpy ndarray, shape = (D, K)):
                Array of Laplace density scale parameters for each mixture component
                to use for initialization
            pi_init (None or numpy ndarray, shape = (K,)):
                Mixture proportions to use for initialization
            step (float):
                Initial step size to use during training
            epochs (int): number of epochs for training
        """
        N, D = X.shape
        K = self.K
        
        # Convert to Tesors
        X_tensor = torch.from_numpy(X)
        # Get DataLoader
        dataset = TensorDataset(X_tensor)
        # Build DataLoader
        tr_loader = DataLoader(dataset, batch_size=64)
        # Model
        model = Mixture(D, K, mu_init, b_init, pi_init)
        
        optimizer = optim.Adam(model.parameters(), lr=step)

        for epoch in range(epochs):
            for X_batch, in tr_loader:
                optimizer.zero_grad()
                
                likelihood = model.forward(X_batch)
                (-likelihood).backward()
                
                optimizer.step()
        
        # Set the parameters
        [mu, b, pi] = model.parameters()

        self.mu = mu.detach().numpy()
        self.b = b.detach().numpy()
        self.pi = pi.detach().numpy()

class Mixture(nn.Module):
    def __init__(self, D, K, mu_init=None, b_init=None, pi_init=None):
        super(Mixture, self).__init__()
        self.K = K
                      
        if mu_init is None:
            mu_init = np.random.rand(D, K)
        
        if b_init is None:
            b_init = np.random.rand(D, K)
        
        if pi_init is None:
            pi_init = np.random.rand(K)
            
        self.mu = nn.Parameter(torch.from_numpy(mu_init))
        self.b = nn.Parameter(torch.from_numpy(b_init))
        self.pi = nn.Parameter(torch.from_numpy(pi_init))
    
    def forward(self, X):
        ''' The marginal log likelihood'''
        mu, b, pi = self.mu, self.b, self.pi
        N, D = X.shape

        pi_norm = pi.softmax(dim=0)

        mgl_likelihood = torch.zeros((N, self.K))
        
        X_nan = torch.isnan(X)
        X[X_nan] = 0

        for k in range(self.K):
            mu_d, b_d = mu[:, k], torch.exp(b[:, k])

            p_xz = 1 / (2 * b_d) * torch.exp(-torch.abs(X - mu_d) / b_d)

            p_xz[X_nan] = 1.

            p_xz_z = p_xz.prod(dim=1) * pi_norm[k]

            mgl_likelihood[:,k] = p_xz_z
        
        mgl_likelihood = torch.log(mgl_likelihood.sum(dim=1)).sum()

        return mgl_likelihood

def main():
    
    data=np.load("../data/data.npz")
    xtr1 = data["xtr1"]
    xtr2 = data["xtr2"]
    xte1 = data["xte1"]
    xte2 = data["xte2"]
    
    np.random.seed(0)
    
    N, D = xtr1.shape
    K = 5

    mm = mixture_model(K=K)
    mm.set_params(np.random.rand(D, K), np.random.rand(D, K), np.random.rand(K))
    mm.fit(xtr2)

if __name__ == '__main__':
    main()