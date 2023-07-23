# Prateek Gulati  
# 12/14/2022  
# CS 7180 Advanced Perception  

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


# matplotlib plot image
def image_plt(imgs, rows=None, cols=None, prompt=None):
#     assert len(imgs) == rows*cols
    fig = plt.figure( figsize=(16,16))
    if cols is None:
        cols=5
        rows=len(imgs)//cols+1
    for i in range(len(imgs)):
        ax = fig.add_subplot(rows, cols, i+1) # this line adds sub-axes
        ax.axis('off')
        ax.set_title(prompt[i],fontsize=10)
#         ...
        ax.imshow(imgs[i]) # t
    plt.tight_layout()

# PIL image grid plot
def image_grid(imgs, rows, cols, prompt=""):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def batch(iterable, size=1):
    l = len(iterable)
    for ndx in range(0, l, size):
        yield iterable[ndx:min(ndx + size, l)]
        

# linear interpolation between two points in latent space
def interpolate_points(p1, p2, s=0,t=1,n_steps=10 ):
    # interpolate ratios between the points
    ratios = torch.linspace(s, t, steps=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return torch.stack(vectors)

# arithmatic subtraction between two points in latent space
def subtract_points(p1, p2, s=0,t=1,n_steps=10 ):
    # interpolate ratios between the points
    ratios = torch.linspace(s, t, steps=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = p1 - ratio * p2
        vectors.append(v)
    return torch.stack(vectors)

# arithmatic substitution between two points in latent space
def substitute_vector(p1, p2, p3, s=0.2,t=0.8,n_steps=10 ):
    # interpolate ratios between the points
    ratios = torch.linspace(s, t, steps=n_steps)
    vectors = list()
    for ratio in ratios:
        v = p1 - ratio * p2 + ratio*p3
        vectors.append(v)
    return torch.stack(vectors)

# spherical linear interpolation between two points in latent space    
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res
    
# spherical interpolation between two points in latent space
def interpolate_points_slerp(p1, p2, s=0,t=1,n_steps=10 ):
    # interpolate ratios between the points
    ratios = torch.linspace(s, t, steps=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        vectors.append(slerp(ratio,p1,p2))
    return torch.stack(vectors)

# normalized interpolation between two points in latent space
def interpolate_points_normalized(p1, p2, s=0,t=1,n_steps=10 ):
    # interpolate ratios between the points
    ratios = torch.linspace(s, t, steps=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        v=v/torch.sqrt(torch.square(1-ratio)+torch.square(ratio))
        vectors.append(v)
    return torch.stack(vectors)

# modified arithmatic subtraction between two points in latent space
def subtract_projection(p1, p2, s=0,t=1,n_steps=10 ):
    # interpolate ratios between the points
    ratios = torch.linspace(s, t, steps=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 + ratio) * p1 - ratio * p2
        vectors.append(v)
    return torch.stack(vectors)