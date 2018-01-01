# Generative Adversarial Nets
This [paper](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) is the original start point of the GAN in machine learning community. The first author of this paper is [Ian Goodfellow](https://scholar.google.ca/citations?user=iYN86KEAAAAJ).

## Model
The general structure GAN is formed by two individual multilayer proceptrons which are called generative model and discriminative model. The generative model, ![](https://latex.codecogs.com/gif.latex?G%28z%3B%5Ctheta_%7Bg%7D%29) is used to map an input noise variable z from a prior distribution ![](https://latex.codecogs.com/gif.latex?p_%7Bz%7D) to the data space with probability distribution ![](https://latex.codecogs.com/gif.latex?p_%7Bg%7D). The discriminative model takes input x and outputs a single scalar which is the probability that x is from original data rather than the generative model's distribution ![](https://latex.codecogs.com/gif.latex?p_%7Bg%7D). Therefore, these two models could form a two-player minimax game which could be represented as:

![](https://latex.codecogs.com/gif.latex?%5Cunderset%7BG%7D%7B%5Ctext%7Bmin%7D%7D%5Cunderset%7BD%7D%7B%5Ctext%7Bmax%7D%7D%7EV%28D%2CG%29%20%3D%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bdata%7D%7D%20%5BlogD%28x%29%5D%20&plus;%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bg%7D%7D%20%5Blog%281%20-%20D%28G%28z%29%29%29%5D)

## Heuristic 
In practice, there are several heuristics which is used to make sure the efficiency and convergence.

1. The optimization process put the discriminative model as inner loop and generative model as outer loop. However, to fully optimize the discriminative model in each iteration with non-optimal generative model and limited datasets could lead to overfitting. Therefore, during implementation, GAN put the discriminative model at inner loop and only optimize k steps in each iteration.

2. From the previous minimax game, the gradient for the generative model is calculated from equation ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Ctheta_%7Bg%7D%7D%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bg%7D%7D%20%5Blog%281%20-%20D%28G%28z%3B%5Ctheta_%7Bg%7D%29%29%29%5D). Compared with generative model, the discriminative model is easier to train in the early stage. This will make ![](https://latex.codecogs.com/gif.latex?log%281%20-%20D%28G%28z%3B%5Ctheta_%7Bg%7D%29%29%29) saturate and the gradient is too small to train the generative model. To solve this problem, GAN train the generative model to maximize the dual function ![](https://latex.codecogs.com/gif.latex?logD%28G%28z%3B%5Ctheta_%7Bg%7D%29%29).

## Theoretical
There are two major components in the theoretical analysis of this paper. The first one is the proof of global optimality at ![](https://latex.codecogs.com/gif.latex?p_%7Bg%7D%20%3D%20p_%7Bdata%7D) while the second one is the proof of convergence.

## Global Optimal
For a fixed generative model, the training process of discriminative model is to maximize 

![](https://latex.codecogs.com/gif.latex?%26%20V%28G%2CD%29%20%3D%20%26%20%5Cint_%7Bx%7D%20p_%7Bdata%7D%20%5Ctimes%20log%28D%28x%29%29d_%7Bx%7D%20&plus;%20%5Cint_%7Bz%7D%20p_%7Bz%7D%20%5Ctimes%20log%281%20-%20D%28g%28z%29%29%29d_%7Bz%7D)
![](https://latex.codecogs.com/gif.latex?%3D%20%5Cint_%7Bx%7D%20p_%7Bdata%7D%20%5Ctimes%20log%28D%28x%29%29%20&plus;%20p_%7Bg%7D%20%5Ctimes%20log%281%20-%20D%28x%29%29d_%7Bx%7D)

Therefore, the best discriminative model for a given generative model is

![](https://latex.codecogs.com/gif.latex?D%5E%7B*%7D_%7BG%7D%28x%29%3D%5Cfrac%7Bp_%7Bdata%7D%28x%29%7D%7Bp_%7Bdata%7D%28x%29%20&plus;%20p_%7Bg%7D%28x%29%7D)

During the training process of generative model, it trys to minimize the value of objective function with an optimized discriminative model which maximize this value. The training objective function for generative model could be reformed as

![](https://latex.codecogs.com/gif.latex?C%28G%29%20%3D%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bdata%7D%7D%20%5BlogD%5E%7B*%7D_%7BG%7D%28x%29%5D%20&plus;%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bg%7D%7D%20%5Blog%281%20-%20D%5E%7B*%7D_%7BG%7D%28x%29%29%5D)

![](https://latex.codecogs.com/gif.latex?%3D%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bdata%7D%7D%20%5Blog%20%5Cfrac%7Bp_%7Bdata%7D%28x%29%7D%7Bp_%7Bdata%7D%28x%29%20&plus;%20p_%7Bg%7D%28x%29%7D%5D%20&plus;%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bg%7D%7D%5Blog%20%5Cfrac%7Bp_%7Bg%7D%28x%29%7D%7Bp_%7Bdata%7D%28x%29%20&plus;%20p_%7Bg%7D%28x%29%7D%5D)

By reformulate the expectation, it's easy to get

![](https://latex.codecogs.com/gif.latex?%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bdata%7D%7D%5Blog%20%5Cfrac%7Bp_%7Bdata%7D%28x%29%7D%7Bp_%7Bdata%7D%28x%29&plus;p_%7Bg%7D%28x%29%7D%5D%20%3D%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bdata%7D%7D%5Blog%28%5Cfrac%7B1%7D%7B2%7D%20%5Ctimes%20%5Cfrac%7B2p_%7Bdata%7D%28x%29%7D%7Bp_%7Bdata%7D%28x%29&plus;p_%7Bg%7D%28x%29%7D%29%5D)

![](https://latex.codecogs.com/gif.latex?%3D%20-log2%20&plus;%20%5Cmathbb%7BE%7D_%7Bx%20%5Csim%20p_%7Bdata%7D%7D%5Blog%20%5Cfrac%7B2p_%7Bdata%7D%7D%7Bp_%7Bdata%7D&plus;p_%7Bg%7D%7D%5D)

![](https://latex.codecogs.com/gif.latex?%3D%20-log2%20&plus;%20D_%7BKL%7D%20%28p_%7Bdata%7D%20%7C%7C%20%5Cfrac%7Bp_%7Bdata%7D&plus;p_%7Bg%7D%7D%7B2%7D%29)

Similarly, the objective function could be reformed as

![](https://latex.codecogs.com/gif.latex?C%28G%29%20%3D%20-log4%20&plus;%20D_%7BKL%7D%28p_%7Bdata%7D%20%7C%7C%20%5Cfrac%7Bp_%7Bdata%7D&plus;p_%7Bg%7D%7D%7B2%7D%29%20&plus;%20D_%7BKL%7D%28p_%7Bg%7D%20%7C%7C%20%5Cfrac%7Bp_%7Bdata%7D&plus;p_%7Bg%7D%7D%7B2%7D%29)

![](https://latex.codecogs.com/gif.latex?%3D%20-log4%20&plus;%202%20%5Ctimes%20JSD%28p_%7Bdata%7D%20%7C%7C%20p_%7Bg%7D%29)

Since the Jensen-Shannon divergence is non-negative, it's clear that the global minimum of the generative model's objective function is ![](https://latex.codecogs.com/gif.latex?-log4) when ![](https://latex.codecogs.com/gif.latex?p_%7Bg%7D%20%3D%20p_%7Bdata%7D).

## Convergence
To proof the convergence of the algorithm, the minimax objective function is considered as a function of ![](https://latex.codecogs.com/gif.latex?p_%7Bg%7D). Moreover, the objectivew function is convex in ![](https://latex.codecogs.com/gif.latex?p_%7Bg%7D). One existing conclusion used in this proof is 
> The subderivatives of a supremum of convex functions include the derivative of the function at the point where the maximum is attined.

By optimize the inner loop, it's clear that

![](https://latex.codecogs.com/gif.latex?C%28G%29%20%3D%20%5Cunderset%7BD%7D%7B%5Ctext%7Bmax%7D%7D%20%7E%20V%28G%2C%20D%29%20%7E%7E%7E%7E%7E%7E%7E%20D%5E%7B*%7D%20%3D%20%5Cunderset%7BD%7D%7B%5Ctext%7Barg%20max%7D%7D%20%7E%20V%28G%2C%20D%29)

Therefore, during optimizing the generative model, the calculated gradient could be used to update the generative model and make ![](https://latex.codecogs.com/gif.latex?p_%7Bg%7D) converge to ![](https://latex.codecogs.com/gif.latex?p_%7Bdata%7D).

![](https://latex.codecogs.com/gif.latex?%5Ctext%7Bgradient%7D%20%3D%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Ctheta_%7Bg%7D%7D%20C%28G%29%20%3D%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Ctheta_%7Bg%7D%7D%20V%28G%2C%20D%5E%7B*%7D%29)

## Question
1. For a fixed discriminative model ![](https://latex.codecogs.com/gif.latex?D%5E%7B*%7D), the original minimax function becomes a function of ![](https://latex.codecogs.com/gif.latex?p_%7Bg%7D). Why this minimax function is also convex in ![](https://latex.codecogs.com/gif.latex?p_%7Bg%7D)? 
