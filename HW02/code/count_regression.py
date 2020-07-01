import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class CountRegression:
    """Count regression.

    Arguments:
       lam (float): regaularization parameter lambda
    """
    def __init__(self, lam):
        self.lam = lam
    
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
        n_samples, n_features = X.shape
        init_wb = np.zeros((n_features + 1,))
        
        w_prime, loss_value, _ = fmin_l_bfgs_b(
            func=self.objective,
            fprime=self.objective_grad,
            x0=init_wb,
            args=(X, y),
            disp=10
        )

        print('Min loss value', loss_value)
        self.set_params(w_prime[:-1], w_prime[-1])
        

    def predict(self, X):
        """Predict using the model.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,): predicted values
        """
        n_samples, n_features = X.shape
        
        w, b = self.get_params()
        w = w.reshape((n_features, 1))
        
        # Expected value of geometric dist = 1/p
        linear_comp = X @ w + b
        fx_inv = np.floor(1 + np.exp(-linear_comp))
        
        f = 1 / (1 + np.exp(-linear_comp))
        
        return np.round((1 - f) / f)
        

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
        n_samples, n_features = X.shape
        
        x_biases = np.ones((n_samples, 1))
        X_bias = np.hstack((X, x_biases))
        assert X_bias.shape == (n_samples, n_features + 1)
        
        wb = wb.reshape((n_features + 1, 1))
        linear_component = X_bias @ wb
        assert linear_component.shape == (n_samples, 1)
        
        y = y.reshape((n_samples, 1))
        
        # Old approach
        # (1 + y) log( 1 + exp(linear) ) + y * linear
        # log_likely = (1 + y) * np.log(1 + np.exp(-linear_component)) + (y * linear_component)
        
        # Simplified
        # log(1 + exp(-linear)) + y*log(1 + exp(linear))
        log_likely = np.log(1 + np.exp(-linear_component)) + y * np.log(1 + np.exp(linear_component))
        assert log_likely.shape == (n_samples, 1)
        data_loss = np.sum(log_likely)
        
        reg_loss = self.lam * np.sum(wb ** 2)
        
        loss = data_loss + reg_loss
        
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
        n_samples, n_features = X.shape
        lmbd = self.lam
        w, b = wb[:-1], wb[-1]
        wb = wb.reshape((n_features + 1, 1))
        
        w = w.reshape((n_features, 1))
        y = y.reshape((n_samples, 1))

        ones = np.ones((n_samples, 1))
        X_bias = np.hstack((X, ones))
        
        linear_comp = X_bias @ wb
        assert linear_comp.shape == (n_samples, 1)

        linear_exp = np.exp(linear_comp)
        dL = (y * linear_exp - 1) / (1 + linear_exp)
        assert dL.shape == (n_samples, 1)
        
        db = np.sum(dL) + (lmbd * 2 * b) # (1)
        
        dw = (X.T @ dL)
        assert dw.shape == (n_features, 1)
        dw = dw + (lmbd * 2 * w) # (2)

        dw = dw.reshape(n_features)
        
        dwb = np.hstack((dw, db))
        assert dwb.shape == (n_features + 1,)

        return dwb

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

    data = np.load("data/count_data.npz")
    X_train=data['X_train']
    X_test=data['X_test']
    Y_train=data['Y_train']
    Y_test=data['Y_test']

    #Define and fit model
    cr = CountRegression(1e-4)
    cr.fit(X_train,Y_train)        
    
if __name__ == '__main__':
    main()
