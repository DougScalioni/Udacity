import torch
import numpy as np
from torchvision import datasets, transforms
from model import Network


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])

    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    print('Data loaded.')
    return trainloader, validloader, testloader


def save_model(net, save_dir='', name='checkpoint.pth'):
    if save_dir != '':
        save_dir += '/'
    path = save_dir + name
    check = {'input_size': 224,
             'hidden_units': net.hidden_units,
             'input_classifier': net.classifier_n_inputs,
             'output_size': net.n_outputs,
             'arch': net.architecture,
             'state_dict': net.model.state_dict()}

    torch.save(check, path)


def load_model(path='checkpoint.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load = torch.load(path, map_location=lambda storage, loc: storage)
    arch = load['arch']
    hidden_units = load['hidden_units']
    parameters = load['state_dict']
    net = Network(arch=arch, hidden_units=hidden_units)
    net.model.to(device)
    net.model.load_state_dict(parameters)
    return net


def process_image(image):
    image = image.resize(size=(256, 256))
    image = crop_center(image, 224)
    np_image = np.array(image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image / 255 - mean) / std

    np_image = np_image.transpose(2, 0, 1)

    tensor = torch.tensor([np_image])
    return tensor


def crop_center(image, dim):
    width, height = image.size

    left = (width - dim) / 2
    top = (height - dim) / 2
    right = (width + dim) / 2
    bottom = (height + dim) / 2

    image = image.crop((left, top, right, bottom))
    return image
