import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


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

    def __init__(self, arch='vgg16', hidden_units=[15, 50]):
        self.model = get_model(arch)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_units = hidden_units
        self.n_classes = 102

        for param in self.model.parameters():
            param.requires_grad = False

        classifier_n_inputs = self.model.classifier[0].in_features
        self.model.classifier = self.get_classifier(classifier_n_inputs, hidden_units)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
        self.model.to(device=self.device)

    def get_classifier(self, n_inputs, hidden_units):
        layers = [n_inputs]
        for h in hidden_units:
            layers.append[h]
        layers.append(self.n_classes)
        sequence = [nn.Linear(layers[0], layers[1])]
        for i in range(1, len(layers) - 1):
            sequence.append(nn.ReLU())
            sequence.append(nn.Dropout(p=0.25))
            sequence.append(nn.Linear(layers[i], layers[i + 1]))
        sequence.append(nn.LogSoftmax(dim=1))
        classifier = nn.Sequential(*sequence)
        return classifier
    

model = get_model('vgg16')

print(model.classifier[0].in_features)
