import argparse
import torch
import util
from model import Network

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
                    default=1,
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
n_classes = 102   # number of species to be identified
dropout = 0.25    # dropout ratio
print_every = 10  # validate every 'print_every' training batches

# Use GPU for training:
gpu_enabled = results.gpu


# Training
trainloader, validloader, testloader = util.load_data(data_directory)
net = Network(architecture, hidden_units, learn_rate, n_classes, dropout)
print('Training for', epochs, 'epochs.')
print("Dropout:", dropout, "Learning rate:", learn_rate)

for e in range(epochs):
    steps = 0
    running_loss = 0
    net.model.train()

    for image, label in trainloader:
        image, label = image.to(net.device), label.to(net.device)

        steps += 1

        net.optimizer.zero_grad()
        logps = net.model.forward(image)
        loss = net.criterion(logps, label)
        loss.backward()
        net.optimizer.step()

        running_loss += loss.item()
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            net.model.eval()
            with torch.no_grad():
                for im, lb in validloader:
                    im, lb = im.to(net.device), lb.to(net.device)
                    logps = net.model.forward(im)
                    valid_loss += net.criterion(logps, lb).item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == lb.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {e + 1}/{epochs}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Validation loss: {valid_loss / len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy / len(validloader):.3f}")

            net.model.train()
            running_loss = 0

util.save_model(net, save_directory)
print('Model trained and saved in', save_directory)
