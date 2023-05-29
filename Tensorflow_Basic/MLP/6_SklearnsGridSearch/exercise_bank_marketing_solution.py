# %% -------------------------------------------------------------------------------------------------------------------
# Train a MLP to predict whether a customer will subscribe to a bank after a pre-defined phone call.
# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical

if "bank-additional-full.csv" not in os.listdir(os.getcwd()):
    try:
        os.system("wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip")
        os.system("unzip bank-additional.zip")
        os.system("mv bank-additional/bank-additional-full.csv bank-additional-full.csv")
    except:
        print("There was a problem with the download")
        raise
        # 1. Download bank-additional.zip from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing, unzip it and move
        # bank-additional-full.csv to the current directory.
    if "bank-additional-full.csv" not in os.listdir(os.getcwd()):
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
# 2. Choose the hyper-parameters values you want to test on the GridSearchCV. You should manually try different learning
# rates and number of epochs at least, to narrow down the search before doing the grid search.
LR = [5e-3]
N_NEURONS = [(100, 200, 100)]
N_EPOCHS = [200]
BATCH_SIZE = [512]
DROPOUT = [0.2, 0.3]
ACTIVATION = ["relu"]

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# 3. Read the data and handle missing values and do whatever other pre-processing you want after some optional EDA.
data = pd.read_csv("bank-additional-full.csv", sep=";")
data.replace("?", np.NaN, inplace=True)
print("Number of examples before dropping nans:", len(data))
data["default"].replace("unknown", "?", inplace=True)  # To treat "unknown" for "default" as another category
data.replace("unknown", np.NaN, inplace=True)  # Dropping the rest of unknowns
data.dropna(inplace=True)
print("Number of examples after dropping nans:", len(data))

x, y = pd.get_dummies(data.drop(["y"], axis=1)).values, LabelEncoder().fit_transform(data[["y"]].values.reshape(-1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=SEED, stratify=y)
distrib = np.unique(y_test, return_counts=True)
print("No info rate is", distrib[1][0]/len(y_test), "for y =", distrib[0][0])


# %% -------------------------------------- Training Prep --------------------------------------------------------------
# 4. Define the function that will return the MLP and also the GridSearchCV instance that uses KerasClassifier, and fit.
# Think about which metric you should use for GridSearchCV, although without more info we can't really make a good
# decision in this case. We would need data on the money the bank makes out of each customer they sign up and the amount
# of money it spends on people calling customers.
def construct_model(dropout=0.3,
                    activation='relu',
                    n_neurons=(100, 200, 100),
                    lr=1e-3
                    ):

    mlp = Sequential([
        Dense(n_neurons[0], input_dim=58, activation=activation),
        Dropout(dropout),
        BatchNormalization()
    ])
    for neurons in n_neurons[1:]:
        mlp.add(Dense(neurons, activation=activation))
        mlp.add(Dropout(dropout, seed=SEED))
        mlp.add(BatchNormalization())
    mlp.add(Dense(2, activation="softmax"))
    mlp.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy")

    return mlp


model = GridSearchCV(
    estimator=KerasClassifier(
        build_fn=construct_model
    ),
    scoring="accuracy",
    param_grid={
        'epochs': N_EPOCHS,
        "batch_size": BATCH_SIZE,
        'dropout': DROPOUT,
        "activation": ACTIVATION,
        'n_neurons': N_NEURONS,
        'lr': LR,
    },
    n_jobs=1,
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
    verbose=100
)

# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train)

print("The best parameters are:", model.best_params_)
results = pd.DataFrame(model.cv_results_).drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
                                                'params', 'std_test_score', 'rank_test_score'], axis=1)
results.drop(list(results.columns[results.columns.str.contains("split")]), axis=1, inplace=True)
results.sort_values(by="mean_test_score", ascending=False).to_excel("exercise_experiments.xlsx")
print("Saving refitted best model on the whole training set...")
model.best_estimator_.model.save("mlp_bank_marketing_5.hdf5")

# %% ------------------------------------------ Final test -------------------------------------------------------------
# 5. After you are content with a cross-val performance on the train set, get the final accuracy on the test set.
print("Final test acc:", model.score(x_test, y_test))

# 6. Use the best model's hyperparameters to get the model again, but this time as a regular Sequential model (you
# can use the construct_model function). Split the train set into train and dev, use the train set to train this
# model and perform early stopping on the dev set. Then get the final performance on the test set and compare with 5.
best_params = model.best_estimator_.get_params()
del best_params["batch_size"], best_params["epochs"], best_params["build_fn"]
best_model = construct_model(**best_params)
best_params = model.best_estimator_.get_params()
x_trainn, x_dev, y_trainn, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
y_trainn, y_dev = to_categorical(y_trainn, num_classes=2), to_categorical(y_dev, num_classes=2)
best_model.fit(x_trainn, y_trainn, batch_size=best_params["batch_size"], epochs=best_params["epochs"],
               validation_data=(x_dev, y_dev), callbacks=[ModelCheckpoint("mlp_bank_marketing_6.hdf5",
                                                                          monitor="val_loss", save_best_only=True)])
print(accuracy_score(y_test, np.argmax(best_model.predict(x_test), axis=1)))
print(confusion_matrix(y_test, np.argmax(best_model.predict(x_test), axis=1)))


# 7. Repeat 4. and 5. but this time performing EarlyStopping during the grid search. Use some patience value so that the
# training process is stopped if the val loss does not decrease for some epochs. After that, do 6. with the new best set
# of hyper-parameteres. Compare the two new final test scores with each other and with 5. and 6. Hint: Follow
# https://stackoverflow.com/questions/48127550/early-stopping-with-keras-and-sklearn-gridsearchcv-cross-validation
model = GridSearchCV(
    estimator=KerasClassifier(
        build_fn=construct_model,
        validation_split=0.1
    ),
    scoring="accuracy",
    param_grid={
        'epochs': N_EPOCHS,
        "batch_size": BATCH_SIZE,
        'dropout': DROPOUT,
        "activation": ACTIVATION,
        'n_neurons': N_NEURONS,
        'lr': LR,
    },
    n_jobs=1,
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
    verbose=10
)
fit_params = dict(callbacks=[EarlyStopping(monitor='val_loss', patience=20)])
model.fit(x_train, y_train, **fit_params)
model.best_estimator_.model.save("mlp_bank_marketing_7.hdf5")
print("Final test acc:", model.score(x_test, y_test))

best_params = model.best_estimator_.get_params()
del best_params["batch_size"], best_params["epochs"], best_params["build_fn"], best_params["validation_split"]
best_model = construct_model(**best_params)
best_params = model.best_estimator_.get_params()
x_trainn, x_dev, y_trainn, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
y_trainn, y_dev = to_categorical(y_trainn, num_classes=2), to_categorical(y_dev, num_classes=2)
best_model.fit(x_trainn, y_trainn, batch_size=best_params["batch_size"], epochs=best_params["epochs"],
               validation_data=(x_dev, y_dev), callbacks=[ModelCheckpoint("mlp_bank_marketing_8.hdf5",
                                                                          monitor="val_loss", save_best_only=True)])
best_model = load_model("mlp_bank_marketing_8.hdf5")
print(accuracy_score(y_test, np.argmax(best_model.predict(x_test), axis=1)))
print(confusion_matrix(y_test, np.argmax(best_model.predict(x_test), axis=1)))
