# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from . import config 


class CoNet(tf.Module):
    def __init__(self, num_users, num_items_A, num_items_B, args):
        self.config = tf.ConfigProto()
        if args.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            self.config.gpu_options.allow_growth = True
            
        self.graph = tf.Graph()
                    
        # Model hyperparameters
        self.num_users = num_users
        self.num_items_A, self.num_items_B = num_items_A, num_items_B
        self.emb_size_u = config.emb_size_u
        self.emb_size_v = config.emb_size_v        
        self.emb_size = self.emb_size_u + self.emb_size_v  # Concat        
        self.layers = config.layers
        self.num_layers = len(self.layers)        
        self.init_std = config.init_std        
        self.max_grad_norm = config.max_grad_norm
        self.activation = config.activation                
        self.weights_A_B = config.weights_A_B # Multi-task
        self.cross_layers = config.cross_layers # Multi-task

        assert self.cross_layers > 0 and self.cross_layers < self.num_layers
        assert self.layers[0] == self.emb_size_u + self.emb_size_v

        # Training hyperparameters
        self.loss_fn = config.loss_fn
        self.dropout_rate = args.dropout_rate
        self.optimizer_name = args.optimizer
        self.lr = args.lr

        self._build_model()

    def _build_model(self):
        with self.graph.as_default():
            self._build_inputs()
        
            self._build_params_shared()

            self._build_params_A_specific()
            self._build_model_A_training()

            self._build_params_B_specific()
            self._build_model_B_training()

            self._build_model_joint_training()

    def _build_inputs(self):
        self.input_A = tf.placeholder(tf.int32, [None, 2], name="input")  # (batch_size, 2) user-item pair
        self.label_A = tf.placeholder(tf.float32, [None, 2], name="label") # (batch_size, 2) one-hot, indicating pos or neg
        self.input_B = tf.placeholder(tf.int32, [None, 2], name="input")  # (batch_size, 2) user-item pair
        self.label_B = tf.placeholder(tf.float32, [None, 2], name="label") # (batch_size, 2) one-hot, indicating pos or neg      

    def _build_params_shared(self):
        """ Parameters: shared """
        # 1. Embedding matrices for users in domain A, users in domain B: shared user embedding matrix
        self.U = tf.Variable(tf.random_normal([self.num_users, self.emb_size_u], stddev=self.init_std))  # Sharing user factors
        # 2. Match the dimensions
        self.shared_Hs = dict()
        for h in range(1, self.cross_layers + 1):  # Only cross between
            self.shared_Hs[h] = tf.Variable(tf.random_normal([self.layers[h - 1], self.layers[h]], stddev=self.init_std))

    def _build_params_A_specific(self):
        """ Parameters: specific """
        # 1. Embedding matrices for users in domain A, users in domain B: shared user embedding matrix
        self.V_A = tf.Variable(tf.random_normal([self.num_items_A, self.emb_size_v], stddev=self.init_std))

        # 2. Weights & biases for hidden layers: the input to hidden layers are the merged embedding
        self.weights_A = dict()
        self.biases_A = dict()
        for h in range(1, self.num_layers):  # (num_layers - 1) weights matrix in hidden layers
            self.weights_A[h] = tf.Variable(tf.random_normal([self.layers[h - 1], self.layers[h]], stddev=self.init_std))
            self.biases_A[h] = tf.Variable(tf.random_normal([self.layers[h]], stddev=self.init_std))
        # 3. Output layer: weight and bias
        self.H_A = tf.Variable(tf.random_normal([self.layers[-1], 2], stddev=self.init_std))
        self.b_A = tf.Variable(tf.random_normal([2], stddev=self.init_std))

    def _build_model_A_training(self):
        """ Computational graph: A training only """
        # 1. Input & embedding layer
        user_emb_A = tf.nn.embedding_lookup(self.U, self.input_A[:, 0])  # 3D due to batch
        item_emb_A = tf.nn.embedding_lookup(self.V_A, self.input_A[:, 1])
        ui_emb_A = tf.concat(values=[user_emb_A, item_emb_A], axis=1)  # No info loss, and emb_size = emb_size_u + emb_size_v

        # 2. MLP: hidden layers, http://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/
        self.layer_h_As = dict()
        layer_h_A = tf.reshape(ui_emb_A, [-1, self.emb_size])  # Init: merged embedding
        self.layer_h_As[0] = layer_h_A  # tf.identity(layer_h_A)
        for h in range(1, self.num_layers):  # (num_layers - 1) weights matrix in hidden layers
            layer_h_A = tf.add(tf.matmul(self.layer_h_As[h-1], self.weights_A[h]), self.biases_A[h])
            if self.activation == "ReLU":
                layer_h_A = tf.nn.relu(layer_h_A)
            elif self.activation == "Tanh":
                layer_h_A = tf.nn.tanh(layer_h_A)
            elif self.activation == "Sigmoid":
                layer_h_A = tf.nn.sigmoid(layer_h_A)
            self.layer_h_As[h] = layer_h_A  # tf.identity(layer_h_A)
            # layer_h = tf.layers.dropout(layer_h, rate=self.dropout_rate) https://www.tensorflow.org/get_started/mnist/pros
        # `layer_h` is now the representations of last hidden layer

        # 3. Output layer: dense and linear
        self.logits_A_only = tf.matmul(layer_h_A, self.H_A) + self.b_A

        # Loss and optimization

        if self.loss_fn == "CrossEntropy":
            self.loss_A_only = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_A_only, labels=self.label_A)
        else:
            self.loss_A_only = tf.losses.hinge_loss(logits=self.logits_A_only, labels=self.label_A)
        self.loss_A_only = tf.reduce_mean(self.loss_A_only)

        if self.optimizer_name  == "Adam":
            self.optimizer_A = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_name  == "RMSProp":
            self.optimizer_A = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9)
        elif self.optimizer_name  == "AdaGrad":
            self.optimizer_A = tf.train.AdagradOptimizer(learning_rate=self.lr)
        else:
            self.optimizer_A = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        params = [self.U, self.V_A, self.H_A, self.b_A]
        for h in range(1, self.num_layers):  # Weights/biases in hidden layers
            params.append(self.weights_A[h])
            params.append(self.biases_A[h])
        grads_and_vars = self.optimizer_A.compute_gradients(self.loss_A_only, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars]

        self.train_op_A = self.optimizer_A.apply_gradients(clipped_grads_and_vars)

    def _build_params_B_specific(self):
        """ Parameters: B specific """
        # 1. Embedding matrices for users in domain A, users in domain B: shared user embedding matrix
        self.V_B = tf.Variable(tf.random_normal([self.num_items_B, self.emb_size_v], stddev=self.init_std))

        # 2. Weights & biases for hidden layers: the input to hidden layers are the merged embedding
        self.weights_B = dict()
        self.biases_B = dict()
        for h in range(1, self.num_layers):  # (num_layers - 1) weights matrix in hidden layers
            self.weights_B[h] = tf.Variable(tf.random_normal([self.layers[h-1], self.layers[h]], stddev=self.init_std))
            self.biases_B[h] = tf.Variable(tf.random_normal([self.layers[h]], stddev=self.init_std))
        # 3. Output layer: weight and bias
        self.H_B = tf.Variable(tf.random_normal([self.layers[-1], 2], stddev=self.init_std))
        self.b_B = tf.Variable(tf.random_normal([2], stddev=self.init_std))

    def _build_model_B_training(self):
        """ Computational graph: B training only """
        # 1. Input & embedding layer
        user_emb_B = tf.nn.embedding_lookup(self.U, self.input_B[:,0])
        item_emb_B = tf.nn.embedding_lookup(self.V_B, self.input_B[:,1])
        ui_emb_B = tf.concat(values=[user_emb_B, item_emb_B], axis=1)  # no info loss, and emb_size = emb_size_u + emb_size_v

        # 2. MLP: hidden layers, http://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/
        self.layer_h_Bs = dict()
        layer_h_B = tf.reshape(ui_emb_B, [-1, self.emb_size])  # Init: cmerged embedding
        self.layer_h_Bs[0] = layer_h_B  # tf.identity(layer_h_B)
        for h in range(1, self.num_layers):  # (num_layers - 1) weights matrix in hidden layers
            layer_h_B = tf.add(tf.matmul(self.layer_h_Bs[h-1], self.weights_B[h]), self.biases_B[h])
            if self.activation == "ReLU":
                layer_h_B = tf.nn.relu(layer_h_B)
            elif self.activation == "Tanh":
                layer_h_B = tf.nn.tanh(layer_h_B)                
            elif self.activation == "Sigmoid":
                layer_h_B = tf.nn.sigmoid(layer_h_B)
            self.layer_h_Bs[h] = layer_h_B
            # layer_h = tf.layers.dropout(layer_h, rate=self.dropout_rate) https://www.tensorflow.org/get_started/mnist/pros
        # `layer_h` is now the representations of last hidden layer

        # 3. Output layer: dense and linear
        self.logits_B_only = tf.matmul(layer_h_B, self.H_B) + self.b_B

        # Loss and optimization
        
        if self.loss_fn == "CrossEntropy":
            self.loss_B_only = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_B_only, labels=self.label_B)
        else:
            self.loss_B_only = tf.losses.hinge_loss(logits=self.logits_B_only, labels=self.label_B)
        self.loss_B_only = tf.reduce_mean(self.loss_B_only)

        if self.optimizer_name  == "Adam":
            self.optimizer_B = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_name  == "RMSProp":
            self.optimizer_B = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9)
        elif self.optimizer_name  == "AdaGrad":
            self.optimizer_B = tf.train.AdagradOptimizer(learning_rate=self.lr)
        else:
            self.optimizer_B = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        params = [self.U, self.V_B, self.H_B, self.b_B]
        for h in range(1, self.num_layers):  # Weights/biases in hidden layers
            params.append(self.weights_B[h])
            params.append(self.biases_B[h])
        grads_and_vars = self.optimizer_B.compute_gradients(self.loss_B_only, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars]

        self.train_op_B = self.optimizer_B.apply_gradients(clipped_grads_and_vars)

    def _build_model_joint_training(self):
        """ Computational graph: for joint training """
        # 1. Input & embedding layer
        user_emb_A = tf.nn.embedding_lookup(self.U, self.input_A[:,0])  # 3D due to batch
        user_emb_B = tf.nn.embedding_lookup(self.U, self.input_B[:,0])
        item_emb_A = tf.nn.embedding_lookup(self.V_A, self.input_A[:,1])
        item_emb_B = tf.nn.embedding_lookup(self.V_B, self.input_B[:,1])
        ui_emb_A = tf.concat(values=[user_emb_A, item_emb_A], axis=1)  # no info loss, and emb_size = emb_size_u + emb_size_v
        ui_emb_B = tf.concat(values=[user_emb_B, item_emb_B], axis=1)  # no info loss, and emb_size = emb_size_u + emb_size_v

        # 2. MLP: hidden layers, http://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/,
        # cross computation between A network and B network
        self.layer_h_As = dict()
        layer_h_A = tf.reshape(ui_emb_A, [-1, self.emb_size])  # Init: merged embedding
        self.layer_h_As[0] = layer_h_A
        self.layer_h_Bs = dict()
        layer_h_B = tf.reshape(ui_emb_B, [-1, self.emb_size])  # Init: merged embedding
        self.layer_h_Bs[0] = layer_h_B
        for h in range(1, self.num_layers):  # (num_layers - 1) weights matrix in hidden layers
            # 1) A-specific: o_A^t+1 = (W_A^t a_A^t + b_A^t) + a_B^t
            layer_h_A = tf.add(tf.matmul(self.layer_h_As[h-1], self.weights_A[h]), self.biases_A[h])
            if h <= self.cross_layers:
                layer_h_A = tf.add(layer_h_A, tf.matmul(self.layer_h_Bs[h-1], self.shared_Hs[h]))
            if self.activation == "ReLU":
                layer_h_A = tf.nn.relu(layer_h_A)
            elif self.activation == "Tanh":
                layer_h_A = tf.nn.tanh(layer_h_A)                    
            elif self.activation == "Sigmoid":
                layer_h_A = tf.nn.sigmoid(layer_h_A)
            self.layer_h_As[h] = layer_h_A
            # 2) B-specific:  o_B^t+1 = (W_B^t a_B^t + b_B^t) + a_A^t
            layer_h_B = tf.add(tf.matmul(self.layer_h_Bs[h-1], self.weights_B[h]), self.biases_B[h])
            if h <= self.cross_layers:
                layer_h_B = tf.add(layer_h_B, tf.matmul(self.layer_h_As[h-1], self.shared_Hs[h]))
            if self.activation == "ReLU":
                layer_h_B = tf.nn.relu(layer_h_B)
            elif self.activation == "Tanh":
                layer_h_B = tf.nn.tanh(layer_h_B)                   
            elif self.activation == "Sigmoid":
                layer_h_B = tf.nn.sigmoid(layer_h_B)
            self.layer_h_Bs[h] = layer_h_B

        # 3. Output layer: dense and linear
        self.logits_A_joint = tf.matmul(layer_h_A, self.H_A) + self.b_A
        self.logits_B_joint = tf.matmul(layer_h_B, self.H_B) + self.b_B

        # Loss and optimization

        if self.loss_fn == "CrossEntropy":
            self.loss_A_joint = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_A_joint, labels=self.label_A)
            self.loss_B_joint = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_B_joint, labels=self.label_B)
        else:
            self.loss_A_joint = tf.losses.hinge_loss(logits=self.logits_A_joint, labels=self.label_A)
            self.loss_B_joint = tf.losses.hinge_loss(logits=self.logits_B_joint, labels=self.label_B)
        self.loss_A_joint = tf.reduce_mean(self.loss_A_joint)
        self.loss_B_joint = tf.reduce_mean(self.loss_B_joint)

        if self.optimizer_name == "Adam":
            self.optimizer_joint = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.optimizer_name == "RMSProp":
            self.optimizer_joint = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9)
        elif self.optimizer_name == "AdaGrad":
            self.optimizer_joint = tf.train.AdagradOptimizer(learning_rate=self.lr)
        else:
            self.optimizer_joint = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        params = [self.U, self.V_A, self.H_A, self.b_A, self.V_B, self.H_B, self.b_B]
        for h in range(1, self.num_layers):  # Weights/biases in hidden layers
            params.append(self.weights_A[h])
            params.append(self.biases_A[h])
            params.append(self.weights_B[h])
            params.append(self.biases_B[h])
        for h in range(1, self.cross_layers+1):
            params.append(self.shared_Hs[h])  # Only cross these layers
        self.loss_joint = self.weights_A_B[0] * self.loss_A_joint + self.weights_A_B[1] * self.loss_B_joint
        grads_and_vars = self.optimizer_joint.compute_gradients(self.loss_joint, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars]

        self.train_op_joint = self.optimizer_joint.apply_gradients(clipped_grads_and_vars)



                

