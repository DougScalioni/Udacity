import torch
from torch import nn
from torch import optim
from torchvision import models


def get_model(arch):
    """
    :rtype: nn.Module
    """
    model = None
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        print("Architecture not available.")
    return model


class Network:

    def __init__(self, arch='vgg16', hidden_units=[4096, 1024], learn_rate=0.003, outputs=102, dropout=0.25, gpu=False):
        self.architecture = arch
        self.model = get_model(self.architecture)
        print('Pre-trained model', self.architecture, "loaded.")
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        self.hidden_units = hidden_units
        self.n_outputs = outputs

        for param in self.model.parameters():
            param.requires_grad = False

        self.classifier_n_inputs = self.model.classifier[0].in_features
        self.model.classifier = self.get_classifier(self.classifier_n_inputs, hidden_units, dropout)
        print('Classifier adapted.')
        print('With', self.classifier_n_inputs, 'inputs, hidden layers of', hidden_units, ',', outputs, 'outputs.')
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), learn_rate)
        self.model.to(device=self.device)

    def get_classifier(self, n_inputs, hidden_units, dropout):
        layers = [n_inputs]
        for h in hidden_units:
            layers.append(h)
        layers.append(self.n_outputs)
        sequence = [nn.Linear(layers[0], layers[1])]
        for i in range(1, len(layers) - 1):
            sequence.append(nn.ReLU())
            sequence.append(nn.Dropout(p=dropout))
            sequence.append(nn.Linear(layers[i], layers[i + 1]))
        sequence.append(nn.LogSoftmax(dim=1))
        classifier = nn.Sequential(*sequence)
        return classifier
