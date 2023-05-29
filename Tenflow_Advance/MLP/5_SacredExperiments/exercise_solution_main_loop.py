# We need to use the dataset_to_use variable for some stuff. Because we are running this script from
# exercise_solution_run_experiment.py, then we can use __main__ as a reference to exercise_solution_run_experiment.py
from __main__ import dataset_to_use
from exercise_solution_get_results import get_results
import os
import json
import tensorflow as tf
import pandas as pd
import numpy as np
from sacred import Experiment


# Different name for each dataset
ex = Experiment('exercise_{}_mlp'.format(dataset_to_use))


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
@ex.config
def my_config():
    random_seed = 42
    lr = 1e-3
    neurons_per_layer = (100, 200, 100)
    n_epochs = 2
    batch_size = 512
    dropout = 0.2
    # 2. Add an option on main_loop.py so that the best model of each run is saved, instead of saving only the best
    # model out of all the runs. The best place to save it is on the folders with the ids for each run.
    save_each = True
    # 3. Add an option to load a model from a run and continue training after changing some of the hyper-parameters,
    # like the learning rate, the optimizer or the number of epochs.
    continue_training_run = None


# %% -------------------------------------- MLP Class ------------------------------------------------------------------
class MLP(tf.keras.Model):
    """ MLP with len(neurons_per_layer) hidden layers """
    def __init__(self, neurons_per_layer, dropout):
        super(MLP, self).__init__()
        self.model_layers = [tf.keras.Sequential([tf.keras.layers.Dense(neurons_per_layer[0], input_shape=(784,))])]
        self.model_layers[0].add(tf.keras.layers.ReLU())
        self.model_layers[0].add(tf.keras.layers.BatchNormalization())
        for neurons in neurons_per_layer[1:]:
            self.model_layers.append(
                tf.keras.Sequential([
                    tf.keras.layers.Dense(neurons),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.BatchNormalization()])
            )
        self.model_layers.append(tf.keras.layers.Dense(10))
        self.drop = dropout
        self.training = True

    def call(self, x):
        for layer in self.model_layers[:-1]:
            x = tf.nn.dropout(layer(x, training=self.training), self.drop)
        return self.model_layers[-1](x)


@ex.automain
def my_main(random_seed, lr, neurons_per_layer, n_epochs, batch_size, dropout, save_each, continue_training_run):

    # %% --------------------------------------- Set-Up ----------------------------------------------------------------
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    # %% -------------------------------------- Data Prep --------------------------------------------------------------
    if dataset_to_use == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = tf.reshape(x_train, (len(x_train), -1)), tf.reshape(x_test, (len(x_test), -1))
    x_train, x_test = tf.dtypes.cast(x_train, tf.float32), tf.dtypes.cast(x_test, tf.float32)
    y_train, y_test = tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)

    # %% -------------------------------------- Training Prep ----------------------------------------------------------
    model = MLP(neurons_per_layer, dropout)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train(x, y):
        model.training = True
        model.drop = dropout
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = criterion(y, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(y, logits)

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def eval(x, y):
        model.training = False
        model.drop = 0
        logits = model(x)
        loss = criterion(y, logits)
        test_loss(loss)
        test_accuracy(y, logits)

    # %% -------------------------------------- Training Loop ----------------------------------------------------------
    if continue_training_run:
        print("\n ------------ Continuing run number {} on run number {} on {} ---------------".format(
            continue_training_run, ex.current_run._id, dataset_to_use))
        with open(os.getcwd() + "/exercise_{}_mlp_runs/{}/config.json".format(dataset_to_use, continue_training_run),
                  "r") as s:
            config_old = json.load(s)
        print("The old config was:", config_old)
        with open(os.getcwd() + "/exercise_{}_mlp_runs/{}/config.json".format(dataset_to_use, ex.current_run._id),
                  "r") as s:
            config_new = json.load(s)
        print("The new config is:", config_new)
        # Updates all the variables with names equal to the keys of the config_new dict with the corresponding values
        locals().update(config_new)
        try:
            results_so_far = pd.read_excel(os.getcwd() + "/exercise_{}_experiments.xlsx".format(dataset_to_use))
            loss_test_best = results_so_far[results_so_far["experiment_id"] == int(continue_training_run)]["test loss"].values.item()
        except:
            print("There was an error getting the best result from previous run ({})."
                  " Will save the best model out of this run".format(continue_training_run))
            loss_test_best = 1000
        model = tf.saved_model.load(os.getcwd() + "/exercise_{}_mlp_runs/{}/exercise_mlp_{}/".format(
            dataset_to_use, continue_training_run, dataset_to_use))
    else:
        print("\n ------------ Doing run number {} on {} with configuration ---------------".format(ex.current_run._id,
                                                                                                    dataset_to_use))
        print(ex.current_run.config)
        if not save_each:
            try:
                get_results(dataset_to_use)
                results_so_far = pd.read_excel(os.getcwd() + "/exercise_solution_{}_experiments.xlsx".format(dataset_to_use))
                loss_test_best = min(results_so_far["test loss"].values)
            except:
                loss_test_best = 1000
                print("No results so far, will save the best model out of this run")
        else:
            loss_test_best = 1000
    best_epoch, loss_best, acc_best = 0, 1000, 0

    print("Starting training loop...")
    for epoch in range(n_epochs):

        for batch in range(len(x_train) // batch_size + 1):
            inds = slice(batch * batch_size, (batch + 1) * batch_size)
            train(x_train[inds], y_train[inds])

        eval(x_test, y_test)
        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, train_loss.result(), train_accuracy.result() * 100, test_loss.result(),
                                        test_accuracy.result() * 100))

        if test_loss.result().numpy() < loss_test_best:
            if save_each:
                tf.saved_model.save(model, os.getcwd() + '/exercise_{}_mlp_runs/{}/exercise_mlp_{}/'.format(
                    dataset_to_use, ex.current_run._id, dataset_to_use))
            else:
                tf.saved_model.save(model, os.getcwd() + '/exercise_mlp_{}/'.format(dataset_to_use))
            print("A new model has been saved!")
            loss_test_best = test_loss.result().numpy()
        if test_loss.result().numpy() < loss_best:
            best_epoch, loss_best, acc_best = epoch, test_loss.result().numpy(), test_accuracy.result().numpy()

        ex.log_scalar("training loss", train_loss.result().numpy(), epoch)
        ex.log_scalar("training acc", train_accuracy.result().numpy(), epoch)
        ex.log_scalar("testing loss", test_loss.result().numpy(), epoch)
        ex.log_scalar("testing acc", test_accuracy.result().numpy(), epoch)
        ex.info["epoch"], ex.info["test loss"], ex.info["test acc"] = best_epoch, loss_best, acc_best

        train_loss.reset_states(); train_accuracy.reset_states(); test_loss.reset_states(); test_accuracy.reset_states()
