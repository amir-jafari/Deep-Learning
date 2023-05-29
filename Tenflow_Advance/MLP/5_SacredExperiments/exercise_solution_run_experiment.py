# %% -------------------------------------------------------------------------------------------------------------------

# 1. Imagine you want to experiment with the same model on two datasets, and you want to keep two experiments files.
# Figure out a way to do this by running run_experiment.py only once, i.e, by running this file the model should be
# trained on both datasets, and the results of the experiments for each dataset should be kept track of separately.
# Use mnist and fashionmnist.

from sacred.observers import FileStorageObserver
from exercise_solution_get_results import get_results
for dataset_to_use in ["mnist", "fashionmnist"]:

    from exercise_solution_main_loop import ex  # Because we need ex to change, we import it inside the for loop
    import sys  # However, Python keeps track of the imported modules on sys.modules["exercise_solution_main_loop"]
    del sys.modules["exercise_solution_main_loop"]  # So if we don't delete it, it won't be imported again
    del sys  # Deletes sys because it's dangerous...
    # Creates or loads an observer with a different name for each dataset
    ex.observers.append(FileStorageObserver.create('exercise_{}_mlp_runs'.format(dataset_to_use)))

    ex.run(config_updates={"lr": 1e-4
                           }
           )
    get_results(dataset_to_use)
    ex.run(config_updates={"neurons_per_layer": (300, 100),
                           "batch_size": 256
                           }
           )
    get_results(dataset_to_use)
    ex.run(config_updates={"dropout": 0.4
                           }
           )

# 3. Add an option to load a model from a run and continue training after changing some of the hyper-parameters, like
# the learning rate, the optimizer or the number of epochs.
dataset_to_use = "fashionmnist"
from exercise_solution_main_loop import ex
import sys
del sys.modules["exercise_solution_main_loop"]
del sys
ex.observers.append(FileStorageObserver.create('exercise_{}_mlp_runs'.format(dataset_to_use)))
ex.run(config_updates={"lr": 1e-3,
                       "continue_training_run": 1
                       }
       )
get_results(dataset_to_use)
