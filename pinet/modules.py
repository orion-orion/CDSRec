# -*- coding: utf-8 -*-
import tensorflow as tf


class FilterCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, member_embedding, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None):
        super(FilterCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._member_embedding = member_embedding
        self._activation = activation or tf.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer 
        
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units    
    
    def call(self, inputs, state): # inputs=[batch_size,hidden_size+hidden_size] , state=[batch_size,self._num_units] 
        inputs_A, inputs_T = tf.split(inputs, num_or_size_splits=2, axis=1) # inputs_A=[batch_size,hidden_size]，inputs_T=[batch_size,hidden_size]
        if self._kernel_initializer is None:
            self._kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        if self._bias_initializer is None:
            self._bias_initializer = tf.constant_initializer(1.0)       
        with tf.variable_scope('gate'): # sigmoid([i_A|i_T|s_(t-1)]*[W_fA;W_fT;U_fS]+emb*V_f+b_f)
            self.W_f = tf.get_variable(dtype=tf.float32, name='W_f', shape=[inputs.get_shape()[-1].value+state.get_shape()[-1].value, self._num_units], initializer=self._kernel_initializer)         
            self.V_f = tf.get_variable(dtype=tf.float32, name='V_f', shape=[self._member_embedding.get_shape()[-1].value, self._num_units], initializer=self._kernel_initializer)
            self.b_f = tf.get_variable(dtype=tf.float32, name='b_f', shape=[self._num_units,], initializer=self._bias_initializer)
            u = tf.matmul(self._member_embedding, self.V_f) # u=[num_members,self._num_units]
            f = tf.concat([inputs, state], axis=-1) # f=[batch_size,hidden_size+hidden_size+self._num_units]
            f = tf.matmul(f, self.W_f) # f=[batch_size,self._num_items]
            f = f+self.b_f # f=[batch_size,self._num_items]
            f = tf.expand_dims(f, axis=1) # f=[batch_size,1,self._num_items]
            f = tf.tile(f, [1,u.get_shape()[0].value,1]) # f=[batch_size,num_members,self._num_items]
            f = f+u # f=[batch_size,num_members,self._num_items]
            f = tf.sigmoid(f)
        with tf.variable_scope('candidate'): # tanh([i_A|s_(t-1)]*[W_sA;U_sS]+emb*V_s+b_s)
            self.W_s = tf.get_variable(dtype=tf.float32, name='W_s', shape=[inputs_A.get_shape()[-1].value+state.get_shape()[-1].value, self._num_units], initializer=self._kernel_initializer) 
            self.V_s = tf.get_variable(dtype=tf.float32, name='V_s', shape=[self._member_embedding.get_shape()[-1].value, self._num_units], initializer=self._kernel_initializer)
            self.b_s = tf.get_variable(dtype=tf.float32, name='b_s', shape=[self._num_units,], initializer=self._bias_initializer)
            _u = tf.matmul(self._member_embedding, self.V_s) # _u=[num_members,self._num_units]
            _s = tf.concat([inputs_A, state], axis=-1) # _s=[batch_size,hidden_size+self._num_units]
            _s = tf.matmul(_s, self.W_s) # _s=[batch_size,self._num_items]
            _s = _s+self.b_s # _s=[batch_size,self._num_items]
            _s = tf.expand_dims(_s, axis=1) # _s=[batch_size,1,self._num_items]
            _s = tf.tile(_s, [1,_u.get_shape()[0].value,1]) # _s=[batch_size,num_members,self._num_items]
            _s = _s+u # _s=[batch_size,num_members,self._num_items]
            _s = self._activation(_s)
        state = tf.expand_dims(state, axis=1) # state=[batch_size,1,self._num_units]
        state = tf.tile(state, [1,self._member_embedding.get_shape()[0].value,1]) # state=[batch_size,num_members,self._num_items]
        new_s = f*state+(1-f)*_s # new_s=[batch_size,num_members,self._num_items]
        new_s = tf.reduce_mean(new_s, axis=1) # new_s=[batch_size,self._num_items]
        return new_s, new_s