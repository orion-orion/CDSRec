# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from .modules import FilterCell
from . import config

class PINet(tf.Module):
    def __init__(self, num_items_A, num_items_B, args):
        self.config = tf.ConfigProto()
        if args.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            self.config.gpu_options.allow_growth = True
            
        self.graph = tf.Graph()

        # Model hyperparameters
        self.num_items_A = num_items_A
        self.num_items_B = num_items_B
        self.num_members = config.num_members
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        # Training hyperparameters
        self.dropout_rate = args.dropout_rate
        self.optimizer_name = args.optimizer
        self.lr = args.lr

        self._build_model()        

    def _build_model(self):
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self._build_inputs()
            with tf.name_scope('encoder_A'):                
                encoder_output_A, encoder_state_A = self._build_encoder_A()                           
            with tf.name_scope('encoder_B'):                
                encoder_output_B, encoder_state_B = self._build_encoder_B()
            with tf.name_scope('filter_B'):              
                filter_output_B, _ = self._build_filter_B(encoder_output_A, encoder_output_B)
            with tf.name_scope('transfer_B'):     
                _, transfer_state_B = self._build_transfer_B(filter_output_B)
            with tf.name_scope('prediction_A'):            
                self.logits_A = self._build_prediction_A(transfer_state_B, encoder_state_A)
            with tf.name_scope('filter_A'):                
                filter_output_A, _ = self._build_filter_A(encoder_output_B, encoder_output_A)
            with tf.name_scope('transfer_A'):                
                _, transfer_state_A = self._build_transfer_A(filter_output_A)
            with tf.name_scope('prediction_B'):                
                self.logits_B = self._build_prediction_B(transfer_state_A, encoder_state_B)
            with tf.name_scope('loss'):                
                self.loss = self._build_loss(self.ground_truth_A, self.logits_A, self.ground_truth_B, self.logits_B)
            with tf.name_scope('train_op'):                
                self.train_op = self._build_train_op(self.loss, self.lr)
        
    def _build_inputs(self):
        self.seq_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_A')
        self.seq_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_B')
        self.len_A = tf.placeholder(dtype=tf.int32, shape=[None,], name='len_A')
        self.len_B = tf.placeholder(dtype=tf.int32, shape=[None,], name='len_B')
        self.pos_A = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='pos_A')
        self.pos_B = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='pos_B')
        self.ground_truth_A = tf.placeholder(dtype=tf.int32, shape=[None,], name='ground_truth_A') 
        self.ground_truth_B = tf.placeholder(dtype=tf.int32, shape=[None,], name='ground_truth_B')
    
    @staticmethod
    def get_gru_cell(hidden_size, dropout_rate):
        gru_cell = tf.contrib.rnn.GRUCell(hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=1 - dropout_rate, output_keep_prob=1 - dropout_rate, state_keep_prob=1 - dropout_rate)  
        return gru_cell
    
    @staticmethod
    def get_filter_cell(hidden_size, member_embedding, dropout_rate):
        filter_cell = FilterCell(hidden_size, member_embedding)
        filter_cell = tf.contrib.rnn.DropoutWrapper(filter_cell, input_keep_prob=1 - dropout_rate, output_keep_prob=1 - dropout_rate, state_keep_prob=1 - dropout_rate)  
        return filter_cell

    def _build_encoder_A(self):
        with tf.variable_scope('encoder_A'):
            embedding_matrix_A = tf.get_variable(dtype=tf.float32, name='embedding_matrix_A', shape=[self.num_items_A, self.embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(embedding_matrix_A)
            embedded_seq_A = tf.nn.embedding_lookup(embedding_matrix_A, self.seq_A) # embedded_seq_A=[batch_size,timestamp_A,embedding_size]
            encoder_cell_A = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(self.hidden_size, self.dropout_rate) for _ in range(self.num_layers)])
            encoder_output_A, encoder_state_A = tf.nn.dynamic_rnn(encoder_cell_A, embedded_seq_A, sequence_length=self.len_A, dtype=tf.float32) # encoder_output_A=[batch_size,timestamp_A,hidden_size], encoder_state_A=([batch_size,hidden_size]*num_layers)       
            print(encoder_output_A)
            print(encoder_state_A)
        return encoder_output_A, encoder_state_A
    
    def _build_encoder_B(self):
        with tf.variable_scope('encoder_B'):
            embedding_matrix_B = tf.get_variable(dtype=tf.float32, name='embedding_matrix_B', shape=[self.num_items_B, self.embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))   
            print(embedding_matrix_B)
            embedded_seq_B = tf.nn.embedding_lookup(embedding_matrix_B, self.seq_B) # embedded_seq_B=[batch_size,timestamp_B,embedding_size]
            encoder_cell_B = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(self.hidden_size, self.dropout_rate) for _ in range(self.num_layers)])
            encoder_output_B, encoder_state_B = tf.nn.dynamic_rnn(encoder_cell_B, embedded_seq_B, sequence_length=self.len_B, dtype=tf.float32) # encoder_output_B=[batch_size,timestamp_B,hidden_size], encoder_state_B=([batch_size,hidden_size]*num_layers)    
            print(encoder_output_B)
            print(encoder_state_B)
        return encoder_output_B, encoder_state_B
    
    def _build_filter_B(self, encoder_output_A, encoder_output_B):
        with tf.variable_scope('filter_B'):
            zero_state = tf.zeros(dtype=tf.float32, shape=(tf.shape(encoder_output_A)[0],1,tf.shape(encoder_output_A)[-1])) # zero_state=[batch_size,1,hidden_size]                                 
            print(zero_state)
            encoder_output = tf.concat([zero_state, encoder_output_A], axis=1) # encoder_output=[batch_size,timestamp_A+1,hidden_size]
            print(encoder_output)
            select_output_A = tf.gather_nd(encoder_output, self.pos_B) # select_output_A=[batch_size,timestamp_B,hidden_size]
            print(select_output_A)
            filter_input_B = tf.concat([encoder_output_B, select_output_A], axis=-1) # filter_input_B=[batch_size,timestamp_B,hidden_size+hidden_size]
            print(filter_input_B)
            member_embedding_B = tf.get_variable(dtype=tf.float32, name='member_embedding_B', shape=[self.num_members, self.embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(member_embedding_B)
            filter_cell_B = tf.contrib.rnn.MultiRNNCell([self.get_filter_cell(self.hidden_size, member_embedding_B, self.dropout_rate) for _ in range(self.num_layers)])
            filter_output_B, filter_state_B = tf.nn.dynamic_rnn(filter_cell_B, filter_input_B, sequence_length=self.len_B, dtype=tf.float32) # filter_output_B=[batch_size,timestamp_B,hidden_size]，filter_state_B=[batch_size,hidden_size]            
            print(filter_output_B)
            print(filter_state_B)
        return filter_output_B, filter_state_B
        
    def _build_transfer_B(self, filter_output_B):
        with tf.variable_scope('transfer_B'):
            transfer_cell_B = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(self.hidden_size, self.dropout_rate) for _ in range(self.num_layers)])
            transfer_output_B, transfer_state_B = tf.nn.dynamic_rnn(transfer_cell_B, filter_output_B, sequence_length=self.len_B, dtype=tf.float32) # transfer_output_B=[batch_size,timestamp_B,hidden_size], transfer_state_B=([batch_size,hidden_size]*num_layers)     
            print(transfer_output_B)
            print(transfer_state_B)
        return transfer_output_B, transfer_state_B
    
    def _build_prediction_A(self, transfer_state_B, encoder_state_A):
        with tf.variable_scope('prediction_A'):
            concat_output = tf.concat([transfer_state_B[-1], encoder_state_A[-1]], axis=-1)                                                      
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output, rate=self.dropout_rate) # concat_output=[batch_size,hidden_size+hidden_size]
            logits_A = tf.layers.dense(concat_output, self.num_items_A, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) # logits_A=[batch_size,num_items_A]
            print(logits_A)
        return logits_A
    
    def _build_filter_A(self, encoder_output_B, encoder_output_A):
        with tf.variable_scope('filter_A'):
            zero_state = tf.zeros(dtype=tf.float32, shape=(tf.shape(encoder_output_B)[0], 1, tf.shape(encoder_output_B)[-1])) # zero_state=[batch_size,1,hidden_size]                                     
            print(zero_state)
            encoder_output = tf.concat([zero_state, encoder_output_B], axis=1) # encoder_output=[batch_size,timestamp_B+1,hidden_size]
            print(encoder_output)
            select_output_B = tf.gather_nd(encoder_output, self.pos_A) # select_output_B=[batch_size,timestamp_A,hidden_size]
            print(select_output_B)
            filter_input_A = tf.concat([encoder_output_A, select_output_B], axis=-1) # filter_input_A=[batch_size,timestamp_A,hidden_size+hidden_size]
            print(filter_input_A)
            member_embedding_A = tf.get_variable(dtype=tf.float32, name='member_embedding_A', shape=[self.num_members, self.embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(member_embedding_A)
            filter_cell_A = tf.contrib.rnn.MultiRNNCell([self.get_filter_cell(self.hidden_size, member_embedding_A, self.dropout_rate) for _ in range(self.num_layers)])
            filter_output_A, filter_state_A = tf.nn.dynamic_rnn(filter_cell_A, filter_input_A, sequence_length=self.len_A, dtype=tf.float32) # filter_output_A=[batch_size,timestamp_A,hidden_size]，filter_state_A=[batch_size,hidden_size]            
            print(filter_output_A)
            print(filter_state_A)
        return filter_output_A, filter_state_A
        
    def _build_transfer_A(self, filter_output_A):
        with tf.variable_scope('transfer_A'):
            transfer_cell_A = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(self.hidden_size, self.dropout_rate) for _ in range(self.num_layers)])
            transfer_output_A, transfer_state_A = tf.nn.dynamic_rnn(transfer_cell_A, filter_output_A, sequence_length=self.len_A, dtype=tf.float32) # transfer_output_A=[batch_size,timestamp_A,hidden_size], transfer_state_A=([batch_size,hidden_size]*num_layers)     
            print(transfer_output_A)
            print(transfer_state_A)
        return transfer_output_A, transfer_state_A
    
    def _build_prediction_B(self, transfer_state_A, encoder_state_B):
        with tf.variable_scope('prediction_B'):
            concat_output = tf.concat([transfer_state_A[-1], encoder_state_B[-1]], axis=-1)
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output, rate=self.dropout_rate) # concat_output=[batch_size,hidden_size+hidden_size]
            logits_B = tf.layers.dense(concat_output, self.num_items_B, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) # logits_B=[batch_size,num_items_B]
            print(logits_B)
        return logits_B
    
    def _build_loss(self, ground_truth_A, logits_A, ground_truth_B, logits_B):
        with tf.name_scope("loss"):        
            loss_A = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ground_truth_A, logits=logits_A)
            loss_A = tf.reduce_mean(loss_A, name='loss_A')
            loss_B = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ground_truth_B, logits=logits_B)
            loss_B = tf.reduce_mean(loss_B, name='loss_B')
            loss = loss_A + loss_B
        return loss
    
    def _build_train_op(self, loss, lr):
        with tf.name_scope("optimizer"):            
            if self.optimizer_name == "Adam":            
                self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)                     
            elif self.optimizer_name == "RMSProp":
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.9)
            elif self.optimizer_name == "AdaGrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)                
              
            grads_and_vars = self.optimizer.compute_gradients(loss)
            clipped_grads_and_vars = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads_and_vars if grad is not None]

            train_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        return train_op