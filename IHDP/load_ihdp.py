import os
import numpy as np
import pandas as pd

def load_ihdp(i_exp = 1, setting = 'B'):

    if 'IHDP' not in os.getcwd():
        os.chdir('IHDP')

    if setting == 'A':

        fname = "A/ihdp_npci_1-100.train.npz"

        data_in = np.load(fname)
        data = {"X": data_in["x"], "t": data_in["t"], "y": data_in["yf"]}

        data["mu0"] = data_in["mu0"]
        data["mu1"] = data_in["mu1"]

        D = data

        D_exp = {}
        D_exp["X"] = D["X"][:, :, i_exp - 1]
        D_exp["t"] = D["t"][:, i_exp - 1: i_exp]
        D_exp["y"] = D["y"][:, i_exp - 1: i_exp]

        D_exp["mu0"] = D["mu0"][:, i_exp - 1: i_exp]
        D_exp["mu1"] = D["mu1"][:, i_exp - 1: i_exp]

        D_train = D_exp

        fname_test = "A/ihdp_npci_1-100.test.npz"
        data_in_test = np.load(fname_test)
        data_test = {"X": data_in_test["x"], "t": data_in_test["t"], "y": data_in_test["yf"]}
        data_test["mu0"] = data_in_test["mu0"]
        data_test["mu1"] = data_in_test["mu1"]

        D_test = {}
        D_test["X"] = data_test["X"][:, :, i_exp - 1]
        D_test["t"] = data_test["t"][:, i_exp - 1: i_exp]
        D_test["y"] = data_test["y"][:, i_exp - 1: i_exp]

        D_test["mu0"] = data_test["mu0"][:, i_exp - 1: i_exp]
        D_test["mu1"] = data_test["mu1"][:, i_exp - 1: i_exp]

        # convert D_train and D_test in dataframes and concatenate them
        D_train = pd.DataFrame(np.concatenate((D_train["X"], D_train["t"], D_train["y"], D_train["mu0"], D_train["mu1"]), axis = 1),
                                columns = ["x" + str(i) for i in range(1, 26)] + ["t"] + ["y"] + ["mu0"] + ["mu1"])
        D_test = pd.DataFrame(np.concatenate((D_test["X"], D_test["t"], D_test["y"], D_test["mu0"], D_test["mu1"]), axis = 1),
                                columns = ["x" + str(i) for i in range(1, 26)] + ["t"] + ["y"] + ["mu0"] + ["mu1"])
        D = pd.concat([D_train, D_test], axis = 0)

        return D

    elif setting == 'B':

        train_data = pd.read_csv(f'B/ihdp_npci_train_{i_exp}.csv')
        test_data = pd.read_csv(f'B/ihdp_npci_test_{i_exp}.csv')

        D = pd.concat([train_data, test_data], axis = 0)

        #rename y_factual as y and treatment as t
        D = D.rename(columns = {'y_factual': 'y', 'treatment': 't'})

        return D
    else:
        print('Invalid setting. Please choose between A and B.')
        return None

