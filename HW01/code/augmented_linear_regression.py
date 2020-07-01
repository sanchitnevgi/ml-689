import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn import linear_model

class AugmentedLinearRegression:
    """Augmented linear regression.

    Arguments:
        delta (float): the trade-off parameter of the loss 
    """
    def __init__(self, delta):
        self.delta = delta

        self.w = None
        self.b = None

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Real-valued output vector for training.
        
        Notes: This function must set member variables such that a subsequent call
        to get_params or predict uses the learned parameters, overwriting 
        any parameter values previously set by calling set_params.
        
        """
        N, D = X.shape
        
        init_W = np.zeros((D + 1,))

        w_prime, loss_value, _ = fmin_l_bfgs_b(
            func=self.objective, 
            fprime=self.objective_grad,
            x0=init_W, 
            args=(X, y), 
            disp=10
        )

        print('Final Loss value:', loss_value)
        
        w, b = w_prime[:-1], w_prime[-1]
        assert w.shape == (D,)
        
        self.set_params(w, b)

    def predict(self, X):
        """Predict using the linear model.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,): predicted values
        """
        print(self.w, self.b)
        return np.dot(X, self.w) + self.b


    def objective(self, wb, X, y):
        """Compute the objective function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            objective (float):
                the objective function evaluated on wb=[w,b] and the data X,y..
        """
        # Loss function
        delta_sq = self.delta ** 2
        
        # Add [X0 = 1] column
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((X, ones)) # shape: (N, 2)

        y_pred = X_b @ wb # shape (N,)

        squared_error = (y - y_pred) ** 2
        
        loss = delta_sq * (np.sqrt(1 + squared_error / delta_sq) - 1)

        loss = np.sum(loss)

        return loss

    def objective_grad(self, wb, X, y):
        """Compute the derivative of the objective function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            objective_grad (ndarray, shape = (n_features + 1,)):
                derivative of the objective function with respect to wb=[w,b].
        """
        N, D = X.shape

        # Add [X0 = 1] column
        X_b = np.hstack((X, np.ones((N, 1)))) # shape: (N, 2)
        
        wb = wb.reshape((D+1, 1))
        y_pred = X_b @ wb
        assert y_pred.shape == (N,1)

        pred_diff = y_pred - y.reshape((N, 1))
        assert pred_diff.shape == (N,1)
        
        diff_by_delta = (pred_diff ** 2) / (self.delta ** 2)
        
        multiplier = pred_diff / np.sqrt(1 + diff_by_delta)

        dW = np.sum(multiplier * X)
        db = np.sum(multiplier)

        dWb = np.array([dW, db])        
        assert dWb.shape == (D+1,)
        
        return dWb

    def get_params(self):
        """Get learned parameters for the model. Assumed to be stored in
           self.w, self.b.

        Returns:
            A tuple (w,b) where w is the learned coefficients (ndarray, shape = (n_features,))
            and b is the learned bias (float).
        """
        return self.w, self.b

    def set_params(self, w, b):
        """Set the parameters of the model. When called, this
           function sets the model parameters tha are used
           to make predictions. Assumes parameters are stored in
           self.w, self.b.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficient prior
            b (float): bias prior
        """
        self.w = w
        self.b = b


def main():

    np.random.seed(0)
    train_X = np.load('../data/q3_train_X.npy')
    train_y = np.load('../data/q3_train_y.npy')

    lr = AugmentedLinearRegression(delta=1)
    lr.fit(train_X, train_y)

if __name__ == '__main__':
    main()
