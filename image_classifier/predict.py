import argparse
from PIL import Image
import torch

import util


parser = argparse.ArgumentParser()

parser.add_argument('image', action="store",
                    dest='image',
                    type=str)

parser.add_argument('checkpoint', action="store",
                    dest='checkpoint',
                    type=str)

parser.add_argument('--top_k', '-k',
                    action='store',
                    dest='top_k',
                    type=int,
                    default=5)

parser.add_argument('--category_names', '-c',
                    action='store',
                    dest='category_names',
                    type=str,
                    default='cat_to_name.json')

parser.add_argument('--gpu', '-g',
                    action='store_true',
                    dest='gpu',
                    default=False)

results = parser.parse_args()

# Inputs:
image = results.image
checkpoint = results.checkpoint

# Return top K most likely classes:
top_k = results.top_k

# Use a mapping of categories to real names:
category_names = results.category_names

# Use GPU for inference:
gpu_enabled = results.gpu

net = util.load_model(checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() and gpu_enabled else "cpu")

im = Image.open(image)
im = util.process_image(im)

net.model = net.model.double()
im = im.to(device)
logps = net.model(im)
ps = torch.exp(logps)
top_p, top_class = ps.topk(top_k)

index = []
probabilities = []
for i in range(len(top_class[0])):
    ind = str(top_class[0][i].item())
    index.append(util.correct_index[ind])

for j in range(len(top_p[0])):
    prob = top_p[0][j].item()
    probabilities.append(prob)


labels = []
for ind in index:
    labels.append(util.cat_to_name[ind])

for i in range(top_k):
    print(index[i], '-', labels[i], '-', probabilities[i])

