import os
import tensorflow as tf
from .modules import embedding, multihead_attention, normalize, feedforward
from . import config 

class TiSASRec(tf.Module):
    def __init__(self, num_items, max_seq_len, args, reuse=tf.AUTO_REUSE):
        self.config = tf.ConfigProto()
        if args.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            self.config.gpu_options.allow_growth = True
            
        self.graph = tf.Graph()

        # Model hyperparameters
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_units = config.hidden_units
        self.num_blocks = config.num_blocks
        self.num_heads = config.num_heads
        self.pad_int = args.pad_int        

        # Training hyperparameters
        self.l2_emb = config.l2_emb
        self.dropout_rate = args.dropout_rate
        self.optimizer_name = args.optimizer
        self.lr = args.lr
        self.reuse = reuse
        
        self._build_model()

    def _build_model(self):
        with self.graph.as_default():
            self.is_training = tf.placeholder(tf.bool, shape=()) # Boolean. Controller of mechanism for dropout.
            self.input_seq = tf.placeholder(tf.int32, shape=(None, None))
            # (batch_size, seq_len, seq_len)            
            self.time_matrix = tf.placeholder(tf.int32, shape=(None, None, None))
            self.pos_samples = tf.placeholder(tf.int32, shape=(None, None))
            self.neg_samples = tf.placeholder(tf.int32, shape=(None, None))
            #  Mask the padding item (i.e., 0)
            # (batch_size, seq_len, 1)
            mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, self.pad_int)), -1)
            
            # A context manager for defining ops that creates variables (layers).
            with tf.variable_scope("SASRec", reuse=self.reuse):
                # 1. Item embedding
                # sequence embedding, item embedding table                                   
                self.seq, item_emb_table = embedding(self.input_seq,
                                                    vocab_size=self.num_items, # num_items
                                                    num_units=self.hidden_units,
                                                    zero_pad=True,
                                                    scale=True,
                                                    l2_reg=self.l2_emb,
                                                    scope="input_embeddings",
                                                    with_t=True,
                                                    reuse=self.reuse
                                                    )
                                                    
                # 2. Absolute position Encoding
                                                    
                # tf.range(tf.shape(self.input_seq)[1]) will get [0, 1, ..., seq_len - 1]
                # after tf.expand_dims(..., 0) its shape will be changed to (1, seq_len).
                # To replicating it as many times as (bach_size, 1)，finally get (batch_size, seq_len).
                # Note that each element of the embedding layer input is a position between [0, max_seq_len),
                # so the vocab_size is max_seq_len                          
                # (batch_size, seq_len)                
                absolute_pos_K = embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                    vocab_size=self.max_seq_len,
                    num_units=self.hidden_units,
                    zero_pad=False,
                    scale=False,
                    l2_reg=self.l2_emb,
                    scope="abs_pos_K",
                    reuse=self.reuse,
                    with_t=False
                )
                # (batch_size, seq_len)
                absolute_pos_V = embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                    vocab_size=self.max_seq_len,
                    num_units=self.hidden_units,
                    zero_pad=False,
                    scale=False,
                    l2_reg=self.l2_emb,
                    scope="abs_pos_V",
                    reuse=self.reuse,
                    with_t=False
                    )
                
                # 3. Relative time interval Encoding

                # The vocab_size is max time_span + 1, i.e., max_seq_len 
                # (batch_size, seq_len)
                time_matrix_emb_K = embedding(
                    self.time_matrix,
                    vocab_size=self.max_seq_len,
                    num_units=self.hidden_units,
                    zero_pad=False,
                    scale=False,
                    l2_reg=self.l2_emb,
                    scope="dec_time_K",
                    reuse=self.reuse,
                    with_t=False
                ) 
                # (batch_size, seq_len)             
                time_matrix_emb_V = embedding(
                    self.time_matrix,
                    vocab_size=self.max_seq_len,
                    num_units=self.hidden_units,
                    zero_pad=False,
                    scale=False,
                    l2_reg=self.l2_emb,
                    scope="dec_time_V",
                    reuse=self.reuse,
                    with_t=False
                )
            
                # Dropout
                self.seq = tf.layers.dropout(self.seq,
                                            rate=self.dropout_rate,
                                            training=tf.convert_to_tensor(self.is_training))
                # Mask the padding item                          
                self.seq *= mask
                
                time_matrix_emb_K = tf.layers.dropout(time_matrix_emb_K,
                                            rate=self.dropout_rate,
                                            training=tf.convert_to_tensor(self.is_training))
                time_matrix_emb_V = tf.layers.dropout(time_matrix_emb_V,
                                            rate=self.dropout_rate,
                                            training=tf.convert_to_tensor(self.is_training))
                absolute_pos_K = tf.layers.dropout(absolute_pos_K,
                                            rate=self.dropout_rate,
                                            training=tf.convert_to_tensor(self.is_training))
                absolute_pos_V = tf.layers.dropout(absolute_pos_V,
                                            rate=self.dropout_rate,
                                            training=tf.convert_to_tensor(self.is_training))
                
                # Build stacking self-attention blocks
                for i in range(self.num_blocks):
                    with tf.variable_scope("num_blocks_%d" % i):

                        # Self-attention
                        self.seq = multihead_attention(queries=normalize(self.seq),
                                                    keys=self.seq,
                                                    time_matrix_K=time_matrix_emb_K,
                                                    time_matrix_V=time_matrix_emb_V,
                                                    absolute_pos_K=absolute_pos_K,
                                                    absolute_pos_V=absolute_pos_V,
                                                    num_units=self.hidden_units,
                                                    num_heads=self.num_heads,
                                                    dropout_rate=self.dropout_rate,
                                                    is_training=self.is_training,
                                                    causality=True,
                                                    scope="self_attention",
                                                    )

                        # Feed forward
                        self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                                            dropout_rate=self.dropout_rate, is_training=self.is_training)
                        
                        # Mask the padding item
                        self.seq *= mask

                self.seq = normalize(self.seq)

            # 4. Prediction layer

            # The representation of the sequence
            # (batch_size * seq_len, hidden_size) 
            seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * tf.shape(self.input_seq)[1], self.hidden_units])
                       
            # Positive samples (each positive sample at position t is the ground truth next item for position t)
            pos_samples = tf.reshape(self.pos_samples, [tf.shape(self.input_seq)[0] * tf.shape(self.input_seq)[1]]) # (batch_size * seq_len)
            # Negative samples
            neg_samples = tf.reshape(self.neg_samples, [tf.shape(self.input_seq)[0] * tf.shape(self.input_seq)[1]]) # (batch_size * seq_len)
            # Look up the embeddings of the positive samples
            pos_emb = tf.nn.embedding_lookup(item_emb_table, pos_samples) # (batch_size * seq_len, hidden_size) 
            # Look up the embeddings of the negative samples
            neg_emb = tf.nn.embedding_lookup(item_emb_table, neg_samples) # (batch_size * seq_len, hidden_size) 

            # For train, compute the inner product of the representation of the sequence at position t
            # and its corresponding positive/negative item embedding in the table,
            # and the result is the prediction logits of the next item(at each position t)
            # (batch_size*seq_len, hidden_size) * (batch_size * seq_len, hidden_size) 
            self.pos_samples_logits = tf.reduce_sum(pos_emb * seq_emb, -1) # (batch_size * seq_len) 
            self.neg_samples_logits = tf.reduce_sum(neg_emb * seq_emb, -1) # (batch_size * seq_len) 

            # For test, compute the inner product of the representation of the sequence at position t
            # and each item embedding in the table, and the result is the prediction logits of the next item(at each position t).
            # (batch_size*seq_len, hidden_size) × (hidden_size, num_items)
            self.test_logits = tf.matmul(seq_emb, tf.transpose(item_emb_table)) # (batch_size * seq_len, num_items)
            # (batch_size, seq_len, num_items)
            self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], tf.shape(self.input_seq)[1], self.num_items])
            # The prediction logits of the next item of the last item (i.e., the next item of the entire sequence)
            self.test_logits = self.test_logits[:, -1, :] # (batch_size, num_items), Note that: not (batch_size, 1, num_items)
            

            # Loss and optimization

            # Train with cross-entropy loss with negative sampling
            # Ignore padding items (default 0)
            is_target = tf.reshape(tf.to_float(tf.not_equal(pos_samples, 0)), [tf.shape(self.input_seq)[0] * tf.shape(self.input_seq)[1]])                                                                                                                                                                               
            self.loss = tf.reduce_sum(
                - tf.log(tf.sigmoid(self.pos_samples_logits) + 1e-24) * is_target -
                tf.log(1 - tf.sigmoid(self.neg_samples_logits) + 1e-24) * is_target
            ) / tf.reduce_sum(is_target)
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss += sum(reg_losses)

            # optimzier
            if self.optimizer_name == "Adam":
                self.optimizer = tf.train.AdamOptimizer(self.lr, beta2=0.98)
            elif self.optimizer_name == "RMSProp":
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9)
            elif self.optimizer_name == "AdaGrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

            self.train_op = self.optimizer.minimize(self.loss)
