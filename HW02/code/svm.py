import numpy as np
import gzip
import pickle


class SVM:
    """SVC with subgradient descent training.

    Arguments:
        C: regularization parameter (default: 1)
        iterations: number of training iterations (default: 500)
    """
    def __init__(self, C=1, iterations=500):
        self.C = C
        self.iterations = iterations
        
        self.w = None
        self.b = None

    def fit(self, X, y):
        """Fit the model using the training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.
        
        Notes: This function must set member variables such that a subsequent call
        to get_params or predict uses the learned parameters, overwriting 
        any parameter values previously set by calling set_params.
        
        """
        n_samples, n_features = X.shape
        f_min, dwb = float('inf'), np.zeros((n_features + 1))

        wb_prime = np.zeros((n_features + 1))

        for i in range(self.iterations):
            sub_grad = self.subgradient(dwb, X, y)

            step_size = 0.002 / np.sqrt(i + 1)

            dwb = dwb - (step_size * sub_grad)

            f = self.objective(dwb, X, y)

            if f < f_min:
                f_min = f
                wb_prime = dwb

        w, b = wb_prime[:-1], wb_prime[-1]
        assert w.shape == (n_features,)
        
        self.set_params(w, b)

    def objective(self, wb, X, y):
        """Compute the objective function for the SVM.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Returns:
            obj (float): value of the objective function evaluated on X and y.
        """
        n_samples, n_features = X.shape

        w = wb[:-1]
        wb = wb.reshape((n_features + 1, 1))
        y = y.reshape((n_samples, 1))

        ones = np.ones((n_samples, 1))
        X_basis = np.hstack((X, ones))
        linear_comp = X_basis @ wb

        hinge_loss = np.maximum(0, 1 - (y * linear_comp))
        assert hinge_loss.shape == (n_samples, 1)

        hinge_loss = self.C * np.sum(hinge_loss)
        reg_loss = np.linalg.norm(w, ord=1)

        loss = hinge_loss + reg_loss

        return loss

    def subgradient(self, wb, X, y):
        """Compute the subgradient of the objective function.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Returns:
            subgrad (ndarray, shape = (n_features+1,)):
                subgradient of the objective function with respect to
                the coefficients wb=[w,b] of the linear model 
        """
        n_samples, n_features = X.shape
        y = y.reshape((n_samples, 1))
        w = wb[:-1].reshape((n_features, 1))
        wb = wb.reshape((n_features + 1, 1))
        
        ones = np.ones((n_samples, 1))
        X_basis = np.hstack((X, ones))
        
        linear = X_basis @ wb
        y_linear = y * linear
        
        hinge_mask = y_linear < 1
        assert hinge_mask.shape == (n_samples, 1)
        
        l1_subgrad = np.zeros_like(w)
        l1_subgrad[w > 0] = 1
        l1_subgrad[w < 0] = -1
        assert l1_subgrad.shape == w.shape
        
        y_masked = y * hinge_mask
        
        # Addition property of subgradient
        dw = -self.C * (X.T @ y_masked) + l1_subgrad
        assert dw.shape == (n_features, 1)

        db = -self.C * np.sum(y_masked)
        
        sub_grad = np.hstack((dw.ravel(), db))
        assert sub_grad.shape == (n_features + 1,)
        
        return sub_grad

    def predict(self, X):
        """Predict class labels for samples in X.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values of -1 or 1.
        """
        w, b = self.get_params()

        n_samples, n_features = X.shape
        w = w.reshape((n_features, 1))

        y_pred = X @ w + b
        assert y_pred.shape == (n_samples, 1)
        
        y_pred = y_pred.ravel()
        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = -1

        return y_pred

    def get_params(self):
        """Get the model parameters.

        Returns:
            w (ndarray, shape = (n_features,)):
                coefficient of the linear model.
            b (float): bias term.
        """
        return self.w, self.b

    def set_params(self, w, b):
        """Set the model parameters.

        Arguments:
            w (ndarray, shape = (n_features,)):
                coefficient of the linear model.
            b (float): bias term.
        """
        self.w = w
        self.b = b

def main():
    np.random.seed(0)

    with gzip.open('../data/svm_data.pkl.gz', 'rb') as f:
        train_X, train_y, test_X, test_y = pickle.load(f)

    clf = SVM(C=1, iterations=1000)
    clf.fit(train_X, train_y)

if __name__ == '__main__':
    main()
