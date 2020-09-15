# nn-memoization

This project aims to study how small can be a neural network that learns noise.
Ideally would compress random data in less than the space it occupies on disk. However since neural networks
are quite "parameter intensive" they usually become usefull after a big number of parameters is used.

Contains two parts.
1. One part is optimization based, here we try different approaches to make a network learn the random data by
backpropagation. Could not make it work on a sufficiently small nn.
2. The second part contains the logic to hardcode the neural network to detect/predict the outputs given the inputs.
