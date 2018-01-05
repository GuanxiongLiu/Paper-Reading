# Learning to Pivot with Adversarial Networks
This [paper](http://papers.nips.cc/paper/6699-learning-to-pivot-with-adversarial-networks.pdf) is published on NIPS 2017. The major work is to train a classifier which is robust to uncertainty features in data through adversarial network.

# Model
The overall model could be seen in the figure below. There are two functional component which are a classifier and a discriminator. The classifier takes an input X and calculate its soft label which is the probability of X belongs to each class. The discriminator takes the calculated soft label as input and outputs the prediction of a pre-defined uncertain parameter S.

![model](https://github.com/GuanxiongLiu/Paper-Reading/blob/Learning-to-Pivot-with-Adversarial-Networks/model.png)

The pre-defined uncertain parameter S could be anything which is hard to be fully specified by training data. By training the classifier in this model, author wants it to classify input X without information from S. In other word, the classification result of X from the classifier is invariant to S.

In order to train the classifier and discriminator simultaneously, author setup a two-player minimax game as the GAN paper. The detailed formulation could be seen as follows:

![minimax1](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Ctheta%7D_%7Bf%7D%2C%7E%5Chat%7B%5Ctheta%7D_%7Bd%7D%20%3D%20%5Ctext%7Barg%7D%7E%5Cunderset%7B%5Ctheta_%7Bf%7D%7D%7B%5Ctext%7Bmin%7D%7D%7E%5Cunderset%7B%5Ctheta_%7Bd%7D%7D%7B%5Ctext%7Bmax%7D%7D%7E%5Cmathcal%7BL%7D_%7Bf%7D%28X%29%20-%20%5Cmathcal%7BL%7D_%7Bd%7D%28f%28X%3B%5Ctheta_%7Bf%7D%29%29)

![minimax2](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bf%7D%28X%29%20%3D%20%5Cmathbb%7BE%7D_%7BX%7D%5Cmathbb%7BE%7D_%7BY%7D%5B-log%28p_%7B%5Ctheta_%7Bf%7D%7D%28y%7Cx%29%29%5D)

![minimax3](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bd%7D%28f%28X%3B%5Ctheta_%7Bf%7D%29%29%20%3D%20%5Cmathbb%7BE%7D_%7BS%7D%5Cmathbb%7BE%7D_%7Bf%28X%3B%5Ctheta_%7Bf%7D%29%7D%5B-log%28p_%7B%5Ctheta_%7Bd%7D%7D%28s%7Cf%28x%3B%5Ctheta_%7Bf%7D%29%29%29%5D)

# Theoretical
In this section, author proof that the classifier and discriminator will finally converge to optimal given its structure has enough capacity and discriminator could reach optimal in each iteration.

Assume a fixed classifier and discriminator reaches its optimal, then

![proof1](https://latex.codecogs.com/gif.latex?p_%7B%5Ctheta_%7Bd%7D%7D%28s%7Cf%28x%3B%5Ctheta_%7Bf%7D%29%29%20%3D%20f%28s%7Cf%28x%3B%5Ctheta_%7Bf%7D%29%29)

![proof2](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bd%7D%28f%28X%3B%5Ctheta_%7Bf%7D%29%29%20%3D%20-%20%5Cint_%7BS%7D%5Cint_%7Bf%28X%3B%5Ctheta_%7Bf%7D%29%7D%20f%28s%2C%20f%28x%3B%5Ctheta_%7Bf%7D%29%29%20%5Ctimes%20log%28f%28s%7Cf%28x%3B%5Ctheta_%7Bf%7D%29%29%29%20d_%7Bs%7Dd_%7Bf%28x%3B%5Ctheta_%7Bf%7D%29%7D)

![proof3](https://latex.codecogs.com/gif.latex?%3D%20H%28S%7Cf%28X%3B%5Ctheta_%7Bf%7D%29%29)

Here, the H is used to denote the conditional entropy.
