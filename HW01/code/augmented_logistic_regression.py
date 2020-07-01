import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn import linear_model

class AugmentedLogisticRegression:
    """Logistic regression with optimized centering.

    Arguments:
        lambda(float): regularization parameter lambda (default: 0)
    """

    def __init__(self, lmbda=0):
        self.reg_param = lmbda  # regularization parameter (lambda)
        
        self.w = self.c = self.b = None

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training output vector. Each entry is either -1 or 1.
        
        Notes: This function must set member variables such that a subsequent call
        to get_params or predict uses the learned parameters, overwriting 
        any parameter values previously set by calling set_params.
        """

        N, D = X.shape
        
        w_init = np.zeros((2*D+1,))
        
        W_prime, loss_value, _ = fmin_l_bfgs_b(
            func=self.objective,
            fprime=self.objective_grad,
            args=(X, y),
            x0= w_init,
            disp=10
        )
        
        print('Final loss value', loss_value)
        
        assert W_prime[:D].shape == (D,)
        
        self.set_params(W_prime[:D], W_prime[D:2*D], W_prime[-1])

    def predict(self, X):
        """Predict class labels for samples in X based on current parameters.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values in {-1, +1}.
        """
        N, D = X.shape
        
        w, c, b = self.get_params()
        mean_X = X - c
        
        y_pred = mean_X @ w + b
        assert y_pred.shape == (N,)
        
        y_pred[y_pred < 0] = -1
        y_pred[y_pred >=0] = 1
        
        return y_pred
        

    def objective(self, wcb, X, y):
        """Compute the learning objective function

        Arguments:
            wcb (ndarray, shape = (2*n_features + 1,)):
                concatenation of the coefficient, centering, and bias parameters
                wcb = [w, c, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            objective (float):
                the objective function evaluated at [w, b, c] and the data X, y.
        """
        N, D = X.shape
        
        w, c, b = wcb[:D], wcb[D:-1], wcb[-1] # shape: (D,) (D,) float
        
        # For each dimension of x_n, subtract c_n. x_ni - c_n
        mean_X = (X - c) # shape: (N, D)
        assert mean_X.shape == (N, D)
        
        wxb = mean_X @ w + b # shape: (N,)
        
        y_wxb = -y * wxb # shape: (N,)
        assert y_wxb.shape == (N,)
        
        data_loss = np.sum(np.log(1 + np.exp(y_wxb)))
        reg_loss = self.reg_param * ((w ** 2).sum() + (c ** 2).sum() + (b ** 2))
        
        loss = data_loss + reg_loss

        return loss

    def objective_grad(self, wcb, X, y):
        """Compute the gradient of the learning objective function

        Arguments:
            wcb (ndarray, shape = (2*n_features + 1,)):
                concatenation of the coefficient, centering, and bias parameters
                wcb = [w, c, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            objective_grad (ndarray, shape = (2*n_features + 1,)):
                gradient of the objective function with respect to [w,c,b].
        """
        lmbd = self.reg_param
        N, D = X.shape
        
        w, c, b = wcb[:D], wcb[D:-1], wcb[-1]
        assert w.shape == (D,) and c.shape == (D,)
        
        mean_X = X - c # shape: (N, D)
        wxb = mean_X @ w + b # shape: (N,)
        assert wxb.shape == (N,)
        
        y_wxb = y * wxb
        assert y_wxb.shape == (N,)
        
        multiplier = (-y / (1 + np.exp(y_wxb))).reshape((N, 1))
        assert multiplier.shape == (N,1)
        
        dw = mean_X * multiplier
        dw = dw.sum(axis=0) + (2 * lmbd * w)
        assert dw.shape == (D,)

        dc = (w * -multiplier.sum()) + (2 * lmbd * c)
        assert dc.shape == (D,)
        
        db = np.sum(multiplier) + (2 * lmbd * b)
        
        dW = np.hstack((dw, dc, db))
        assert dW.shape == (2*D+1,)
        
        return dW

    def get_params(self):
        """Get parameters for the model.

        Returns:
            A tuple (w,c,b) where w is the learned coefficients (ndarray, shape = (n_features,)),
            c  is the learned centering parameters (ndarray, shape = (n_features,)),
            and b is the learned bias (float).
        """
        return self.w, self.c, self.b

    def set_params(self, w, c, b):
        """Set the parameters of the model.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficients
            c (ndarray, shape = (n_features,)): centering parameters
            b (float): bias 
        """
        self.w, self.c, self.b = w, c, b


def main():
    np.random.seed(0)

    train_X = np.load('data/q2_train_X.npy')
    train_y = np.load('data/q2_train_y.npy')
    test_X = np.load('data/q2_test_X.npy')
    test_y = np.load('data/q2_test_y.npy')

    
    lr = AugmentedLogisticRegression(lmbda = 1e-6)
    lr.fit(train_X, train_y)

if __name__ == '__main__':
    main()
