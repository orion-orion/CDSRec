# -*- coding: utf-8 -*-
emb_size_u = 32
emb_size_v = 32
layers = [64,32,16,8] # Layers[0] must equal to emb_size_u + emb_size_v
init_std = 0.01 # Weight initialization std [0.05]
max_grad_norm = 10 # Clip gradients to this norm [50]
activation = "ReLU" # Possible are ReLU, Tanh, Sigmoid
loss_fn = "CrossEntropy" # Possible are CrossEntropy, Hinge
# Multi-task  
weights_A_B = [1, 1] # Weights of each task [0.8,0.2], [0.5,0.5], [1,1]
cross_layers = 2 # Cross between 1st & 2nd, and 2nd & 3rd layers