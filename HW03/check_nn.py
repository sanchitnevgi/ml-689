import argparse
import numpy as np
from code.nn import NN
import gzip
import pickle
from checker_utils import catch_exceptions, send_msg
import time
from pathlib import Path
import os

EVAL     = True
DATA_DIR = Path('data')
TEST_DIR = Path('solution-data')

def create_test_data():
    
    #Set RNG Seed
    seed = int(input("Enter a random seed: "))
    np.random.seed(seed)
    
    #Load data sets
    data = np.load(os.path.join(DATA_DIR, "mnist_rot_train.npz"),allow_pickle=True)
    X_train, y_train, y_angles_train = data["X"], data["labels"], data["angles"]

    #Select a random subset of training data
    samples = 1000
    ind = np.random.permutation(np.arange(y_train.shape[0]))
    train_X = X_train[ind[:samples],:]
    train_y = y_train[ind[:samples]]
    train_y_angles = y_angles_train[ind[:samples]]
    
    #Save data set
    np.save(str(TEST_DIR / 'check_nn_train_X.npy'),train_X)
    np.save(str(TEST_DIR / 'check_nn_train_y.npy'),train_y)
    np.save(str(TEST_DIR / 'check_nn_train_y_angles.npy'), train_y_angles)


def load_data():
    #Load data sets
    train_X = np.load(str(TEST_DIR / 'check_nn_train_X.npy'),allow_pickle=True)
    train_y = np.load(str(TEST_DIR / 'check_nn_train_y.npy'),allow_pickle=True)
    train_y_angles = np.load(str(TEST_DIR / 'check_nn_train_y_angles.npy'),allow_pickle=True) 
    return train_X, train_y, train_y_angles

def create_params():
    train_X, train_y, train_y_angles = load_data()
    model = NN(alpha=.5, epochs=10)
    model.fit(train_X, train_y, train_y_angles)
    params = model.get_params()
    np.save(str(TEST_DIR / 'check_nn_params.npy'), params)
    
def load_params():
    params = np.load(str(TEST_DIR / 'check_nn_params.npy'),allow_pickle=True)
    return params
    
      


def allclose(a, b):
    return np.allclose(a, b, rtol=1e-5, atol=1e-3)


@catch_exceptions
def check_set_get():
    """[Q3] Check the correctness of set() get() loop"""

    params_true = load_params()
    model = NN(alpha=.5, epochs=10)
    model.set_params(params_true)
    params = model.get_params()
     
    param_names = ["w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4", "w5", "b5", "w6", "b6"]
       
    
    if type(params)!=type([]):
        send_msg('Type of params returned by get_params is not list') 
        return
        
    if(len(params)!=12):
        send_msg('Length of params list returned by get_params is not 12') 
        return        
    
    msg=[]
    for i in range(12):
        if not isinstance(params[i], np.ndarray):
            msg.append('%s type is not np.ndarray'%param_names[i])
        else:
            if not params[i].shape == params_true[i].shape:
                msg.append('%s array shape is not correct'%param_names[i])
            else:    
                if not np.allclose(params_true[i], params[i], rtol=1e-10, atol=1e-10):
                    msg.append('%s values are not correct'%param_names[i])
            
    if(len(msg)==0):    
        send_msg('pass') 
    else:
        send_msg(", ".join(msg))
 

@catch_exceptions
def check_objective():
    
    params_ref = load_params()
    X, y, y_angle = load_data()
    model = NN(alpha=.5, epochs=1)
    model.set_params(params_ref)
    obj = model.objective(X, y, y_angle)
    
    try:
        float(obj)
    except TypeError:
        send_msg('Objective value could not be converted to float. Check type.')
        return
    
    if(EVAL): 
        obj_true = np.load(str(TEST_DIR / 'check_nn_obj.npy'),allow_pickle=True)
        send_msg('pass' if np.allclose(obj_true,obj,rtol=1e-5, atol=1e-5) else 'Objective function is not close to reference solution value.')
    else:
        np.save(str(TEST_DIR / 'check_nn_obj.npy'), obj)
        send_msg('pass')
        

@catch_exceptions
def check_predict():
    
    params_ref = load_params()
    X, y, y_angle = load_data()
    model = NN(alpha=.5, epochs=1)
    model.set_params(params_ref)
    y_predict, y_angle_predict = model.predict(X)
    
    if(not (isinstance(y_predict, np.ndarray) and isinstance(y_angle_predict, np.ndarray))):
        send_msg('Prediction output is not of type np.ndarray')
        return
    
    y_predict = y_predict.flatten()
    y_angle_predict = y_angle_predict.flatten()
    
    if(not ( (y_predict.shape == y.shape) and (y_angle_predict.shape == y_angle.shape))):
        send_msg('Prediction output array has the wrong shape')
        return        
    
    if(EVAL): 
        y_predict_true = np.load(str(TEST_DIR / 'check_nn_y_predict.npy'),allow_pickle=True)
        y_angle_predict_true = np.load(str(TEST_DIR / 'check_nn_y_angle_predict.npy'),allow_pickle=True)
        
        msg = []
        if(not np.allclose(y_predict_true,y_predict,rtol=1e-5, atol=1e-5)):
            msg.append("Class predictions are not close to reference solutions")
        if(not np.allclose(y_angle_predict_true,y_angle_predict,rtol=1e-5, atol=1e-5)):
            msg.append("Angle predictions are not close to reference solutions")
        
        if(len(msg)==0):    
            send_msg('pass') 
        else:
            send_msg(", ".join(msg))    
    else:
        np.save(str(TEST_DIR / 'check_nn_y_predict.npy'), y_predict)
        np.save(str(TEST_DIR / 'check_nn_y_angle_predict.npy'),y_angle_predict)
        send_msg('pass')    
    

def main():

    global EVAL    
    np.random.seed(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='target')
    parser.add_argument('-m', '--mode')
    args = parser.parse_args()
    target = args.target
    mode = args.mode

    #If in eval mode, run a test
    #Else run setup for all tests
    if(mode=="eval"):
        EVAL=True
        #Run a test
        if target == 'objective':
            check_objective()
        elif target == 'predict':
            check_predict()
        elif target == 'setget':
            check_set_get()

    elif(mode=="setup"):
        EVAL=False
        import os
        if not os.path.exists(TEST_DIR):
            os.makedirs(TEST_DIR)
        
        check_objective()   
        check_predict()     

if __name__ == '__main__':
    main()
