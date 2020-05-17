import argparse

parser = argparse.ArgumentParser()

parser.add_argument('data_directory', action="store", type=str)

parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    type=str)

parser.add_argument('--arch', action='store',
                    dest='arch',
                    type=str)

parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate',
                    type=float)

parser.add_argument('--hidden_units', action='store',  # list of ints
                    default=[512, 256],
                    dest='hidden_units',
                    nargs='*'
                    )

parser.add_argument('--epochs', action='store',
                    dest='epochs',
                    type=int)

parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu')

results = parser.parse_args()

# Set directory to save checkpoints:
save_directory = results.save_dir

# Choose architecture:
architecture = results.arch

# Set hyper-parameters:
learn_rate = results.learning_rate
hidden_units = results.hidden_units
epochs = results.epochs

# Use GPU for training:
gpu_enabled = results.gpu

print(results)
