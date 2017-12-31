# Generative Adversarial Nets
This [paper](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) is the original start point of the GAN in machine learning community. The first author of this paper is [Ian Goodfellow](https://scholar.google.ca/citations?user=iYN86KEAAAAJ).

## Model
The general structure GAN is formed by two individual multilayer proceptrons which are called generative model and discriminative model. The generative model, ![](https://latex.codecogs.com/gif.latex?G%28z%3B%5Ctheta_%7Bg%7D%29) is used to map an input noise variable z from a prior distribution ![](https://latex.codecogs.com/gif.latex?p_%7Bz%7D) to the data space with probability distribution ![](https://latex.codecogs.com/gif.latex?p_%7Bg%7D). The discriminative model takes input x and outputs a single scalar which is the probability that x is from original data rather than the generative model's distribution ![](https://latex.codecogs.com/gif.latex?p_%7Bg%7D). Therefore, these two models could form a two-player minimax game which could be represented as:

![](https://latex.codecogs.com/gif.latex?%5Cunderset%7BG%7D%7B%5Ctext%7Bmin%7D%7D%5Cunderset%7BD%7D%7B%5Ctext%7Bmax%7D%7D%7EV%28D%2CG%29%20%3D%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bdata%7D%7D%20%5BlogD%28x%29%5D%20&plus;%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bg%7D%7D%20%5Blog%281%20-%20D%28G%28z%29%29%29%5D)

