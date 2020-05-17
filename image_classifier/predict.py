import argparse

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

print(results)
