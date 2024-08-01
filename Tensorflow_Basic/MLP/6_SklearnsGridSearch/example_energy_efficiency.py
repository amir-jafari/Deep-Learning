# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
from time import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

if "ENB2012_data.xlsx" not in os.listdir(os.getcwd()):
    try:
        os.system("wget https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx")
    except:
        print("There was a problem with the download")
        raise
    if "ENB2012_data.xlsx" not in os.listdir(os.getcwd()):
        print("There was a problem with the download")
        import sys
        sys.exit()

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = [5e-4]
N_NEURONS = [(100, 200, 100), (300, 200, 100)]
N_EPOCHS = [2000]
BATCH_SIZE = [512]
DROPOUT = [0.2, 0.3]
ACTIVATION = ["tanh", "relu"]

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
data = pd.read_excel("ENB2012_data.xlsx")  # Reads the dataset into a pandas DataFrame
data.replace("?", np.NaN, inplace=True)  # UCI's nans sometimes come like this
x, y = data.drop(["Y1", "Y2"], axis=1), data[["Y1", "Y2"]]  # Features and target
assert np.all(x.isna().sum() == 0), "There are missing feature values"  # Checks if there are
assert np.all(y.isna().sum() == 0), "There are missing target values"  # any nans
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.3, random_state=SEED)

# %% -------------------------------------- Define Model Function ----------------------------------------------------
def build_model(activation='tanh', dropout=0.3, n_neurons=(100, 200, 100), lr=1e-3):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n_neurons[0], activation=activation, input_dim=x_train.shape[1]))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.BatchNormalization())
    for neurons in n_neurons[1:]:
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout, seed=SEED))
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(2))  # We have two continuous targets
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mean_squared_error")
    return model

# %% -------------------------------------- Grid Search Setup --------------------------------------------------------
param_grid = {
    'epochs': N_EPOCHS,
    'batch_size': BATCH_SIZE,
    'dropout': DROPOUT,
    'n_neurons': N_NEURONS,
    'lr': LR,
    'activation': ACTIVATION
}

model = KerasRegressor(build_fn=build_model, verbose=0)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=SEED), verbose=10, n_jobs=1)

# %% -------------------------------------- Training Loop ----------------------------------------------------------
start = time()
grid_result = grid.fit(x_train, y_train)
print(f"Training time: {time() - start} seconds")

print("The best parameters are:", grid_result.best_params_)

# Get the results into a DataFrame, drop some "irrelevant" columns, sort by best score, and save to a spreadsheet
results = pd.DataFrame(grid_result.cv_results_).drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params', 'std_test_score', 'rank_test_score'], axis=1)
results.drop(list(results.columns[results.columns.str.contains("split")]), axis=1, inplace=True)
results.sort_values(by="mean_test_score", ascending=False).to_excel("example_experiments.xlsx")

# Save the best model
print("Saving refitted best model on the whole training set...")
grid_result.best_estimator_.model.save("mlp_energy.hdf5")

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final test r-squared:", grid_result.score(x_test, y_test))
