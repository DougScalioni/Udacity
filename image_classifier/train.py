import argparse

parser = argparse.ArgumentParser()

parser.add_argument('data_directory', action="store", type=str)

parser.add_argument('--save_dir', '-s',
                    action='store',
                    dest='save_dir',
                    type=str)

parser.add_argument('--arch', '-a',
                    action='store',
                    dest='arch',
                    default='vgg16',
                    type=str)

parser.add_argument('--learning_rate', '-l',
                    action='store',
                    dest='learning_rate',
                    default=0.003,
                    type=float)

parser.add_argument('--hidden_units', '-H',
                    action='store',
                    default=[4096, 1024],
                    dest='hidden_units',
                    nargs='*'
                    )

parser.add_argument('--epochs', '-e',
                    action='store',
                    dest='epochs',
                    default=20,
                    type=int)

parser.add_argument('--gpu', '-g',
                    action='store_true',
                    default=False,
                    dest='gpu')

results = parser.parse_args()

# Inputs:
data_directory = results.data_directory

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
