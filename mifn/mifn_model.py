import os
import tensorflow as tf
from .modules import FilterCell
from . import config


class MIFN(tf.Module):
    def __init__(self, num_items_A, num_items_B, args):
        self.config = tf.ConfigProto()
        if args.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            self.config.gpu_options.allow_growth = True

        self.graph = tf.Graph()
          
        # Model hyperparameters
        self.num_items_A = num_items_A
        self.num_items_B = num_items_B
        self.num_entity_A = self.num_items_A
        self.num_entity_B = self.num_items_B
        self.num_items = self.num_entity_A + self.num_entity_B + config.num_cate
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.num_neighbors = config.num_neighbors
        self.num_h_hops = config.num_h_hops

        # Training hyperparameters
        self.dropout_rate = args.dropout_rate
        self.optimizer_name = args.optimizer
        self.lr = args.lr

        self._build_model()

    def _build_model(self):
        with self.graph.as_default():
            with tf.name_scope('inputs'):            
                self._build_inputs()
            with tf.name_scope('all_encoder'):                
                self._build_all_encoder()
            with tf.name_scope('encoder_A'):                
                encoder_output_A, encoder_state_A = self._build_encoder_A()
            with tf.name_scope('encoder_B'):
                encoder_output_B, encoder_state_B = self._build_encoder_B()
            with tf.name_scope('sequence_transfer_A'):
                filter_output_A, _ = self._build_filter_A(encoder_output_A, encoder_output_B)
                transfer_output_A, transfer_state_A = self._build_transfer_A(filter_output_A)
            with tf.name_scope('sequence_transfer_B'):
                filter_output_B, _ = self._build_filter_B(encoder_output_B, encoder_output_A)
                transfer_output_B, transfer_state_B = self._build_transfer_B(filter_output_B)

            with tf.name_scope('graph_transfer'):
                entity_emb = self._build_graph_gnn(encoder_output_A, encoder_output_B,
                                            transfer_output_A, transfer_output_B)

            with tf.name_scope('prediction_A'):
                self.PG_A, self.PS_A = self._build_switch_A(encoder_state_A, transfer_state_B, entity_emb, self.nei_A_mask)
                s_pred_A = self._build_s_decoder_A(self.num_items_A, encoder_state_A, transfer_state_B, self.dropout_rate)
                g_pred_A, _ = self._build_g_decoder_A(encoder_state_A, entity_emb, self.num_items_A, self.nei_A_mask,
                                                    self.nei_index_A, self.nei_is_in_A)
                self.pred_A = self._build_final_pred_A(self.PG_A, self.PS_A, s_pred_A, g_pred_A)

            with tf.name_scope('prediction_B'):
                self.PG_B, self.PS_B = self._build_switch_B(encoder_state_B, transfer_state_A, entity_emb,self.nei_B_mask)
                s_pred_B = self._build_s_decoder_B(self.num_items_B, encoder_state_B, transfer_state_A, self.dropout_rate)
                g_pred_B, _ = self._build_g_decoder_B(encoder_state_B, entity_emb, self.num_items_B,self.nei_B_mask,
                                                    self.nei_index_B, self.nei_is_in_B)
                self.pred_B = self._build_final_pred_B(self.PG_B, self.PS_B, s_pred_B, g_pred_B)

            with tf.name_scope('loss'):
                self.loss = self._build_loss(self.ground_truth_A, self.pred_A, self.ground_truth_B, self.pred_B)

            with tf.name_scope('optimizer'):
                self.train_op = self._build_train_op(self.loss, self.lr)

    def _build_inputs(self):
        self.seq_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_A') # (batch_size, seq_lenA)
        self.seq_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_B') # (batch_size, seq_lenB)
        self.len_A = tf.placeholder(dtype=tf.int32, shape=[None, ], name='len_A') # (batch_size,)
        self.len_B = tf.placeholder(dtype=tf.int32, shape=[None, ], name='len_B') # (batch_size,)
        self.pos_A = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='pos_A') # (batch_size, seq_lenA, 2)
        self.pos_B = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='pos_B') # (batch_size, seq_lenB, 2)
        self.index_A = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='index_A') # (batch_size, seq_lenA, 2)
        self.index_B = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='index_B') # (batch_size, seq_lenB, 2)
        self.ground_truth_A = tf.placeholder(dtype=tf.int32,shape=[None,],name='ground_truth_A') # (batch_size,)
        self.ground_truth_B = tf.placeholder(dtype=tf.int32,shape=[None,],name='ground_truth_B') # (batch_size,)
        self.ground_A_in_nei = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='ground_A_in_nei') # (batch_size, 1)
        self.ground_B_in_nei = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='ground_B_in_nei') # (batch_size, 1)
        self.adj_1 = tf.placeholder(dtype=tf.float32, shape=[None,self.num_neighbors,self.num_neighbors],name='adj_alb') # (batch_size, nei_num, nei_num)
        self.adj_2 = tf.placeholder(dtype=tf.float32, shape=[None,self.num_neighbors,self.num_neighbors],name='adj_alv') # (batch_size, nei_num, nei_num)
        self.adj_3 = tf.placeholder(dtype=tf.float32, shape=[None,self.num_neighbors,self.num_neighbors],name='adj_bav') # (batch_size, nei_num, nei_num)
        self.adj_4 = tf.placeholder(dtype=tf.float32, shape=[None,self.num_neighbors,self.num_neighbors],name='adj_bt') # (batch_size, nei_num,nei_num)
        self.adj_5 = tf.placeholder(dtype=tf.float32, shape=[None,self.num_neighbors,self.num_neighbors],name='adj_ada') # (batch_size, nei_num, nei_num)
        self.neighbors = tf.placeholder(dtype=tf.int64, shape=[None, self.num_neighbors],name='neighbors') # (batch_size, nei_num)
        self.nei_index_A = tf.placeholder(dtype=tf.int64, shape=[None, self.num_neighbors, 2],name='nei_index_A') # (batch_size, nei_num, 2)
        self.nei_index_B = tf.placeholder(dtype=tf.int64, shape=[None, self.num_neighbors, 2], name='nei_index_B') # (batch_size, nei_num, 2)
        self.nei_A_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.num_neighbors], name='nei_A_mask')  # (batch_size, nei_num)
        self.nei_B_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.num_neighbors], name='nei_B_mask') # (batch_size, nei_num)
        self.nei_is_in_A = tf.placeholder(dtype=tf.float32, shape=[None, self.num_neighbors], name='nei_is_in_A') # (batch_size, nei_num)
        self.nei_is_in_B = tf.placeholder(dtype=tf.float32, shape=[None, self.num_neighbors], name='nei_is_in_B') # (batch_size, nei_num)

        self.nei_L_A_mask = tf.placeholder(dtype=tf.float32, shape=[None, None, self.num_neighbors],
                                        name='nei_L_A_mask') # (batch_size, seq_lenA + seq_lenB, nei_num)
        self.nei_L_T_mask = tf.placeholder(dtype=tf.float32, shape=[None, None, self.num_neighbors],
                                        name='nei_L_T_mask') # (batch_size, seq_lenA + seq_lenB, nei_num)

    @staticmethod
    def get_gru_cell(hidden_size, dropout_rate):
        gru_cell = tf.contrib.rnn.GRUCell(hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=1 - dropout_rate,
                                                 output_keep_prob=1 - dropout_rate,
                                                 state_keep_prob=1 - dropout_rate)
        return gru_cell

    @staticmethod
    def get_filter_cell(hidden_size, dropout_rate):
        filter_cell = FilterCell(hidden_size)
        filter_cell = tf.contrib.rnn.DropoutWrapper(filter_cell, input_keep_prob=1 - dropout_rate, output_keep_prob= 1 - dropout_rate,
                                                    state_keep_prob=1 - dropout_rate)
        return filter_cell

    def _build_all_encoder(self,):
        with tf.variable_scope('all_encoder'):
            self.all_emb_matrix = tf.get_variable(shape=[self.num_items, self.embedding_size], name='item_emb_matrix',
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    def _build_encoder_A(self,):
        with tf.variable_scope('encoder_A'):
            # (batch_size, seq_lenA) -> (batch_size, embedding_size)
            embedd_seq_A = tf.nn.embedding_lookup(self.all_emb_matrix, self.seq_A)
            print(embedd_seq_A)
            encoder_cell_A = tf.nn.rnn_cell.MultiRNNCell([self.get_gru_cell(self.hidden_size, self.dropout_rate) for _ in range(self.num_layers)])
            encoder_output_A, encoder_state_A = tf.nn.dynamic_rnn(encoder_cell_A, embedd_seq_A, sequence_length=self.len_A, dtype=tf.float32)
        return encoder_output_A, encoder_state_A,

    def _build_encoder_B(self,):
        with tf.variable_scope('encoder_B'):
            embedd_seq_B = tf.nn.embedding_lookup(self.all_emb_matrix, self.seq_B)
            print(embedd_seq_B)
            encoder_cell_B = tf.nn.rnn_cell.MultiRNNCell([self.get_gru_cell(self.hidden_size, self.dropout_rate) for _ in range(self.num_layers)])
            encoder_output_B, encoder_state_B = tf.nn.dynamic_rnn(encoder_cell_B, embedd_seq_B, sequence_length=self.len_B, dtype=tf.float32)
        return encoder_output_B, encoder_state_B

    def _build_filter_A(self, encoder_output_A, encoder_output_B,):
        with tf.variable_scope('filter_A'):
            zero_state = tf.zeros(dtype=tf.float32, shape=(tf.shape(encoder_output_A)[0], 1, tf.shape(encoder_output_A)[-1]))
            encoder_output = tf.concat([zero_state, encoder_output_B], axis=1)
            # print(encoder_output) #encoder_output=[batch_size,timestamp_B+1,hidden_size]
            select_output_B = tf.gather_nd(encoder_output,self.pos_A) # 挑出A-output之前的B-item, len还是timestep_A
            # print(select_output_B) #select_output_A=[batch_size,timestamp_A,hidden_size]
            filter_input_A = tf.concat([encoder_output_A, select_output_B], axis=-1)  # filter_input_A=[b,tA,2*h]
            # att = tf.layers.dense(tf.concat([encoder_output_A, select_output_B], axis=-1), units=hidden_size,activation=tf.nn.sigmoid)
            # combined_output_A = att * tf.nn.tanh(encoder_output_A) + (1 - att) * select_output_B
            # print(combined_output_A) #[batch_size,timestamp_A,hidden_size]
            filter_cell_A = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_filter_cell(self.hidden_size, self.dropout_rate) for _ in range(self.num_layers)])
            filter_output_A, filter_state_A = tf.nn.dynamic_rnn(filter_cell_A, filter_input_A, sequence_length=self.len_A,
                                                                dtype=tf.float32)
            # print(filter_output_A)  # filter_output_A=[batch_size,timestamp_A,hidden_size]，
            # print(filter_state_A)  # filter_state_A=[batch_size,hidden_size]
        return filter_output_A, filter_state_A

    def _build_transfer_A(self, filter_output_A,):
        with tf.variable_scope('transfer_A'):
            transfer_cell_A = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_gru_cell(self.hidden_size, self.dropout_rate) for _ in range(self.num_layers)])
            transfer_output_A, transfer_state_A = tf.nn.dynamic_rnn(transfer_cell_A, filter_output_A,
                                                                    sequence_length=self.len_A,dtype=tf.float32)
            # print(transfer_output_A) # transfer_output_A=[batch_size,timestamp_A,hidden_size],
            # print(transfer_state_A)  # transfer_state_A=([batch_size,hidden_size]*num_layers)
        return transfer_output_A, transfer_state_A

    def _build_filter_B(self, encoder_output_B, encoder_output_A):
        with tf.variable_scope('filter_B'):
            zero_state = tf.zeros(dtype=tf.float32,
                                  shape=(tf.shape(encoder_output_B)[0], 1, tf.shape(encoder_output_B)[-1]))
            # print(zero_state)  # zero_state=[batch_size,1,hidden_size]
            encoder_output = tf.concat([zero_state, encoder_output_A], axis=1)
            # print(encoder_output)  # encoder_output=[batch_size,timestamp_B+1,hidden_size]
            select_output_A = tf.gather_nd(encoder_output, self.pos_B)  # 挑出B-output之前的A-item
            # print(select_output_A)  # select_output_B=[batch_size,timestamp_B,hidden_size]
            filter_input_B = tf.concat([encoder_output_B, select_output_A], axis=-1)
            # print(filter_input_B)  # filter_input_B=[batch_size,timestamp_B,hidden_size+hidden_size]

            # att = tf.layers.dense(tf.concat([encoder_output_B, select_output_A], axis=-1), units=hidden_size,
            #                       activation=tf.nn.sigmoid)
            # combined_output_B = att * tf.nn.tanh(encoder_output_B) + (1 - att) * select_output_A
            # print(combined_output_B)  # [batch_size,timestamp_B,hidden_size]
            filter_cell_B = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_filter_cell(self.hidden_size, self.dropout_rate) for _ in range(self.num_layers)])
            filter_output_B, filter_state_B = tf.nn.dynamic_rnn(filter_cell_B, filter_input_B, sequence_length=self.len_B,
                                                                dtype=tf.float32)
            # print(filter_output_B)  # filter_output_B=[batch_size,timestamp_B,hidden_size]，
            # print(filter_state_B)  # filter_state_B=[batch_size,hidden_size]
        return filter_output_B, filter_state_B

    def _build_transfer_B(self, filter_output_B,):
        with tf.variable_scope('transfer_B'):
            transfer_cell_B = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_gru_cell(self.hidden_size, self.dropout_rate) for _ in range(self.num_layers)])
            transfer_output_B, transfer_state_B = tf.nn.dynamic_rnn(transfer_cell_B, filter_output_B,
                                                                    sequence_length=self.len_B,
                                                                    dtype=tf.float32)
            # print(transfer_output_B)  # transfer_output_B=[batch_size,timestamp_B,hidden_size],
            # print(transfer_state_B)   # transfer_state_B=([batch_size,hidden_size]*num_layers)
        return transfer_output_B, transfer_state_B

    def _build_graph_gnn(self, encoder_output_A, encoder_output_B,
                    transfer_output_A, transfer_output_B,):
        with tf.variable_scope('graph_gnn'):
            self.entity_emb = tf.nn.embedding_lookup(self.all_emb_matrix, self.neighbors) # [b,N,e]
            # ----------------- in-domain adj parameter ------------------
            self.W_alb1 = random_weight(self.hidden_size, self.hidden_size, name='W_alb1')
            self.b11 = random_bias(self.hidden_size, name='b11')
            self.W_alv1 = random_weight(self.hidden_size, self.hidden_size, name='W_alv1')
            self.b21 = random_bias(self.hidden_size, name='b21')
            self.W_bav1 = random_weight(self.hidden_size, self.hidden_size, name='W_bav1')
            self.b31 = random_bias(self.hidden_size, name='b31')
            self.W_bt1 = random_weight(self.hidden_size, self.hidden_size, name='W_bt1')
            self.b41 = random_bias(self.hidden_size, name='b41')
            self.W_ada1 = random_weight(self.hidden_size, self.hidden_size, name='W_ada1')
            self.b51 = random_bias(self.hidden_size, name='b51')

            # ------------------- cross-domain adj parameter ------------------
            self.W_alb2 = random_weight(self.hidden_size, self.hidden_size, name='W_alb2')
            self.b12 = random_bias(self.hidden_size, name='b12')
            self.W_alv2 = random_weight(self.hidden_size, self.hidden_size, name='W_alv2')
            self.b22 = random_bias(self.hidden_size, name='b22')
            self.W_bav2 = random_weight(self.hidden_size, self.hidden_size, name='W_bav2')
            self.b32 = random_bias(self.hidden_size, name='b32')
            self.W_bt2 = random_weight(self.hidden_size, self.hidden_size, name='W_bt2')
            self.b42 = random_bias(self.hidden_size, name='b42')
            self.W_ada2 = random_weight(self.hidden_size, self.hidden_size, name='W_ada2')
            self.b52 = random_bias(self.hidden_size, name='b52')

            inputs_A, inputs_A2T,\
            nei_mask_A, nei_mask_T = self.get_ht(encoder_output_A, encoder_output_B,
                                                transfer_output_A, transfer_output_B,
                                                 self.len_A, self.len_B, self.index_A, self.index_B,
                                                 self.nei_L_A_mask, self.nei_L_T_mask)
            inputs_A = tf.tile(tf.expand_dims(inputs_A, axis=1), [1, self.num_neighbors, 1])
            print('input-A:', inputs_A)  # [batch_size, nei_num, hidden_size]
            inputs_A2T = tf.tile(tf.expand_dims(inputs_A2T, axis=1), [1, self.num_neighbors, 1])
            print('input-A2T:', inputs_A2T)  # [batch_size, nei_num, hidden_size]

            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

            # --------------  softmask-A attention weight --------------
            self.W_s_in = random_weight(self.hidden_size, self.hidden_size, name='W_s_in')
            self.W_emb_in = random_weight(self.hidden_size, self.hidden_size, name='W_emb_in')
            self.W_v_in = random_weight(self.hidden_size, 1, name='W_v_in')
            # -------------- softmask-A2T attention weight -------------
            self.W_s_cross = random_weight(self.hidden_size, self.hidden_size, name='W_s_cross')
            self.W_emb_cross = random_weight(self.hidden_size, self.hidden_size, name='W_emb_cross')
            self.W_v_cross = random_weight(self.hidden_size, 1, name='W_v_cross')

            for _ in range(self.num_h_hops):
                softmask_A = self.indomain_attention(inputs_A, nei_mask_A)
                softmask_A2T = self.crossdomain_attention(inputs_A2T, nei_mask_T)
                gcn_emb = self.get_neigh_rep(inputs_A, inputs_A2T,softmask_A, softmask_A2T, nei_mask_A,nei_mask_T)
                print(gcn_emb) # [b, N, 10*h]
                # self.entity_emb = tf.layers.dense(gcn_emb, self.hidden_size,
                #                                   activation=None,
                #                                   kernel_initializer=tf.contrib.layers.xavier_initializer(
                #                                       uniform=False))  # [batch_size, nei_num, hidden_size]
                self.entity_emb = tf.reshape(self.entity_emb, [-1, self.hidden_size])  # [batch_size*nei_num, hidden_size]
                graph_output, self.entity_emb = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(gcn_emb, [-1, 10 * self.hidden_size]), axis=1),
                                      initial_state=self.entity_emb)
                print(graph_output)  # graph_output=[batch_size*nei_num, hidden_size]，
                print(self.entity_emb)  # graph_state=[batch_size*nei_num, hidden_size]*num_layers
            
            self.entity_emb = tf.reshape(self.entity_emb, [-1, self.num_neighbors, self.hidden_size])
            print('after gnn:', self.entity_emb)
        return self.entity_emb

    def get_neigh_rep(self,inputs_A, inputs_A2T,softmask_A, softmask_A2T, nei_mask_A,nei_mask_T):
        with tf.variable_scope('cdgcn', reuse=tf.AUTO_REUSE):

            self.W_c = random_weight(3 * self.hidden_size, self.hidden_size,
                                     name='W_f')
            self.b_c = random_bias(self.hidden_size, name='b_f')
            inputs = tf.concat([inputs_A, inputs_A2T], axis=-1)  # [b, N, 2h]
            f = tf.concat([inputs, self.entity_emb], axis=-1)  # [b, N, 3h]
            f = tf.matmul(f, self.W_c) + self.b_c
            f = tf.sigmoid(f)  # [b, N, h]
            print('cross-gate:', f)  # f=[b, N, h]
            softmask_A = f * softmask_A
            softmask_A2T = (1 - f) * softmask_A2T

            #--------------- in-domain neighbor ------------------
            fin_state_A = tf.reshape(softmask_A, [-1, self.hidden_size])
            self.W1 = random_weight(self.hidden_size, self.hidden_size, name='w1')
            self.b1 = random_bias(self.hidden_size, name='b1')
            fin_state_A = tf.matmul(fin_state_A, self.W1) + self.b1
            print(fin_state_A)  # [b*N, s]

            #---------------- cross-domain neighbor ---------------
            fin_state_A2T = tf.reshape(softmask_A2T, [-1, self.hidden_size])
            self.W2 = random_weight(self.hidden_size, self.hidden_size, name='w2')
            self.b2 = random_bias(self.hidden_size, name='b2')
            fin_state_A2T = tf.matmul(fin_state_A2T, self.W2) + self.b2
            print(fin_state_A2T)  # [b*N, s]

            nei_mask_A = tf.expand_dims(nei_mask_A, -1)  # [b,N,1]
            nei_mask_T = tf.expand_dims(nei_mask_T, -1)  # [b,N,1]
            mask_emb_A = nei_mask_A * self.entity_emb # [b,N,h]
            mask_emb_T = nei_mask_T * self.entity_emb # [b,N,h]
            att_emb_T = tf.reshape(self.mutual_att(mask_emb_T, mask_emb_A), [-1, self.hidden_size])  # [b*N,s]
            fin_state_A2T = tf.add(fin_state_A2T, att_emb_T)
            print(fin_state_A2T)  # [b*N, s]

            # ------------------ in-domain representation --------------
            fin_state_1a = matrix_mutliply(fin_state_A, self.W_alb1, self.b11, self.num_neighbors, self.hidden_size)
            fin_state_2a = matrix_mutliply(fin_state_A, self.W_alv1, self.b21, self.num_neighbors, self.hidden_size)
            fin_state_3a = matrix_mutliply(fin_state_A, self.W_bav1, self.b31, self.num_neighbors, self.hidden_size)
            fin_state_4a = matrix_mutliply(fin_state_A, self.W_bt1, self.b41, self.num_neighbors, self.hidden_size)
            fin_state_5a = matrix_mutliply(fin_state_A, self.W_ada1, self.b51, self.num_neighbors, self.hidden_size)
            all_nei_A = tf.nn.relu(tf.concat([
                                    tf.matmul(self.adj_1, fin_state_1a),
                                    tf.matmul(self.adj_2, fin_state_2a),
                                    tf.matmul(self.adj_3, fin_state_3a),
                                    tf.matmul(self.adj_4, fin_state_4a),
                                    tf.matmul(self.adj_5, fin_state_5a),], axis=-1))
            print(all_nei_A)  # all_nei=[b, N, 5*h]

            ###################### cross-domain representation #################
            fin_state_1t = matrix_mutliply(fin_state_A2T, self.W_alb2, self.b12, self.num_neighbors, self.hidden_size)
            fin_state_2t = matrix_mutliply(fin_state_A2T, self.W_alv2, self.b22, self.num_neighbors, self.hidden_size)
            fin_state_3t = matrix_mutliply(fin_state_A2T, self.W_bav2, self.b32, self.num_neighbors, self.hidden_size)
            fin_state_4t = matrix_mutliply(fin_state_A2T, self.W_bt2, self.b42, self.num_neighbors, self.hidden_size)
            fin_state_5t = matrix_mutliply(fin_state_A2T, self.W_ada2, self.b52, self.num_neighbors, self.hidden_size)
            all_nei_A2T = tf.nn.relu(tf.concat([
                tf.matmul(self.adj_1, fin_state_1t),
                tf.matmul(self.adj_2, fin_state_2t),
                tf.matmul(self.adj_3, fin_state_3t),
                tf.matmul(self.adj_4, fin_state_4t),
                tf.matmul(self.adj_5, fin_state_5t), ], axis=-1))
            print(all_nei_A2T)  # all_nei=[b, N, s*5]
            all_nei = tf.concat([all_nei_A, all_nei_A2T], axis=-1)
            print(all_nei)  # [b, N, 10*s]

        return all_nei

    def get_ht(self,encoder_output_A, encoder_output_B,transfer_output_A, transfer_output_B,
               len_A, len_B, index_A, index_B, nei_L_A_mask, nei_L_T_mask):
        all_len = tf.add(len_A, len_B)
    
        ########## get hAi from encoder ##########
        e1 = tf.scatter_nd(index_A, encoder_output_A, [tf.shape(encoder_output_A)[0],
                                                       tf.shape(encoder_output_A)[1] + tf.shape(encoder_output_B)[1],
                                                       self.hidden_size])
        print(e1)
        e2 = tf.scatter_nd(index_B, encoder_output_B, [tf.shape(encoder_output_A)[0],
                                                       tf.shape(encoder_output_A)[1] + tf.shape(encoder_output_B)[1],
                                                       self.hidden_size])
        print(e2)
        seq_L = e1 + e2  # [b, time_A+time_B, h]
        hA = tf.gather_nd(seq_L, tf.stack([tf.range(tf.shape(encoder_output_A)[0]), all_len-1], axis=1))  # 拿到了最后一步的输入
        print('inputsA:',hA)
        ########## get h(A->B)i from transfer ##########
        e3 = tf.scatter_nd(index_A, transfer_output_A, [tf.shape(transfer_output_A)[0],
                                                        tf.shape(transfer_output_A)[1] + tf.shape(transfer_output_B)[1],
                                                        self.hidden_size])
        print(e3)
        e4 = tf.scatter_nd(index_B, transfer_output_B, [tf.shape(transfer_output_A)[0],
                                                        tf.shape(transfer_output_A)[1] + tf.shape(transfer_output_B)[1],
                                                        self.hidden_size])
        print(e4)
        trans_L = e3 + e4  # [b, time_A+time_B, h]
        hA2T = tf.gather_nd(trans_L, tf.stack([tf.range(tf.shape(encoder_output_A)[0]), all_len-1], axis=1))  # 拿到了最后一步的输入
        print('inputsA2T:', hA2T)
        ########## get mask ###############
        print(tf.shape(nei_L_A_mask)[1])
        nei_A_mask = tf.gather_nd(nei_L_A_mask, tf.stack([tf.range(tf.shape(encoder_output_A)[0]), all_len-1], axis=1))
        print('neiA_mask:', nei_A_mask) # [b, N]
        nei_T_mask = tf.gather_nd(nei_L_T_mask, tf.stack([tf.range(tf.shape(encoder_output_A)[0]), all_len-1], axis=1))
        print('neiA2T_mask:', nei_T_mask) # [b, N]

        return hA, hA2T,nei_A_mask, nei_T_mask

    def indomain_attention(self, item_state, nei_mask):

        with tf.variable_scope('softmask_att_in', reuse=tf.AUTO_REUSE):
            S_it = tf.matmul(item_state, self.W_s_in)
            print('S_it:', S_it)  # [b, N, h]
            S_emb = tf.matmul(self.entity_emb, self.W_emb_in)
            print('S_emb:', S_emb)  # [b, N, h]
            tanh = tf.tanh(S_it + S_emb)
            print("tanh:", tanh)  # [b, N, h]
            s = tf.squeeze(tf.matmul(tanh, self.W_v_in))
            print("s:", s)  # [b, N]
            s_inf_mask = self.mask_softmax(nei_mask, s)
            print(s_inf_mask)  # [b, N]
            score = self.normalize_softmax(s_inf_mask)  # [b, N]
            score = tf.expand_dims(score, axis=-1)
            print('score:', score)  # [b, N, 1]
            softmask = score * self.entity_emb  # [b, N, e]
        return softmask

    def crossdomain_attention(self, item_state, nei_mask):
        with tf.variable_scope('softmask_att_cross', reuse=tf.AUTO_REUSE):
            S_it = tf.matmul(item_state, self.W_s_cross)
            print('S_it:', S_it)  # [b, N, h]
            S_emb = tf.matmul(self.entity_emb, self.W_emb_cross)
            print('S_emb:', S_emb)  # [b, N, h]
            tanh = tf.tanh(S_it + S_emb)
            print("tanh:", tanh)  # [b, N, h]
            s = tf.squeeze(tf.matmul(tanh, self.W_v_cross))
            print("s:", s)  # [b, N]
            nei_mask = tf.reshape(nei_mask, [-1, self.num_neighbors])
            s_inf_mask = self.mask_softmax(nei_mask, s)
            print(s_inf_mask)  # [b, N]
            score = self.normalize_softmax(s_inf_mask)  # [b, N]
            score = tf.expand_dims(score, axis=-1)
            print('score:', score)  # [b, N, 1]
            softmask = score * self.entity_emb  # [b, N, e]
        return softmask

    def mask_softmax(self, seq_mask, scores):
        '''
        to do softmax, assign -inf value for the logits of padding tokens
        '''
        seq_mask = tf.cast(seq_mask, tf.bool)
        scores = tf.reshape(scores, [-1, self.num_neighbors])
        score_mask_values = -1e10 * tf.ones_like(scores, dtype=tf.float32)
        # print("**********************")
        # tf.Print(self.seq_mask, [self.seq_mask])
        # tf.Print(self.scores, [self.scores])
        # tf.Print(self.score_mask_values, [self.score_mask_values])
        seq_mask = tf.reshape(seq_mask, [-1, self.num_neighbors])
        # print("************************")
        return tf.where(seq_mask, scores, score_mask_values)

    def normalize_softmax(self,x):
        max_value = tf.reshape(tf.reduce_max(x, -1), [-1, 1])
        each_ = tf.exp(x - max_value)
        all_ = tf.reshape(tf.reduce_sum(each_, -1), [-1, 1])
        score = each_ / all_
        return score

    def mutual_att(self,hb, hA,):
        hb_ext = tf.expand_dims(hb, axis=2)  # hb_ext=[b,N1,1,h]
        hb_ext = tf.tile(hb_ext, [1, 1, tf.shape(hA)[1], 1])  # hb_ext=[b,N1,N2,h]
        hA_ext = tf.expand_dims(hA, axis=1)  # hA_ext=[b,1,N2,h]
        hA_ext = tf.tile(hA_ext, [1, tf.shape(hb)[1], 1, 1])  # hA_ext=[b,N1,N2,h]
        dot = hb_ext * hA_ext
        # dot = tf.concat([hb_ext, hA_ext, hb_ext * hA_ext], axis=-1)  # dot=[b,N1,N2,h]
        dot = tf.layers.dense(dot, 1, activation=None, use_bias=False)  # dot=[b,N1,N2,1]
        dot = tf.squeeze(dot)  # dot=[b,N1,N2]
        # sum_row = tf.reduce_sum(dot, axis=-1, keep_dims=True)  # sum_row=[b,N1,1]
        # att_hb = sum_row * hb
        # print(att_hb) # [b, N1, h]
        att_hb = tf.matmul(dot, hA)  # [b,N1,h]
        return att_hb

    def _build_switch_A(self, encoder_state_A, transfer_state_B, graph_state, nei_mask):
        with tf.variable_scope('switch_A'):
            graph_rep = tf.reshape(graph_state, [-1, self.num_neighbors, self.hidden_size])
            nei_mask = tf.expand_dims(nei_mask, -1)
            graph_rep = nei_mask * graph_rep
            graph_rep = tf.reduce_sum(graph_rep, axis=1)
            concat_output = tf.concat([encoder_state_A[-1], transfer_state_B[-1], graph_rep], axis=-1)
            linear_switch = tf.layers.Dense(1, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            switch_matrix = linear_switch(concat_output)  # Tensor shape (b, 1)
            PG_A = tf.sigmoid(switch_matrix)
            PS_A = 1 - PG_A
            print('PSA:',PS_A)
            print('PGA:',PG_A)
        return PG_A, PS_A

    def _build_s_decoder_A(self, num_items_A, encoder_state_A, transfer_state_B, dropout_rate):
        with tf.variable_scope('s_predict_A'):
            concat_output = tf.concat([encoder_state_A[-1],transfer_state_B[-1]],axis=-1)
            concat_output = tf.nn.dropout(concat_output, rate=dropout_rate)
            pred_A = tf.layers.dense(concat_output, num_items_A,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            pred_A = tf.nn.softmax(pred_A)
        return pred_A

    def _build_g_decoder_A(self, ht, graph_state, num_items_A, nei_mask, nei_index_A, nei_is_in_A):

        with tf.variable_scope('g_predict_A'):
            self.W_h_a = random_weight(self.hidden_size, self.hidden_size, name='W_h_a')
            self.W_emb_a = random_weight(self.hidden_size, self.hidden_size, name='W_emb_a')
            self.W_v_a = random_weight(self.hidden_size, 1, name='W_v_a')
            graph_state = tf.reshape(graph_state, [-1, self.num_neighbors, self.hidden_size])  # [b, N, h]
            nei_mask = tf.expand_dims(nei_mask, -1)
            graph_state = nei_mask * graph_state
            att = self.g_decode_attention_A(ht[-1], graph_state, nei_is_in_A)
            print(att)  # [b, N]
            g_pred_A = tf.scatter_nd(nei_index_A, att, [tf.shape(graph_state)[0], num_items_A])
            print(g_pred_A)  # [b, num_item_A]
        return g_pred_A, att

    def g_decode_attention_A(self, ht, repre, mask):
        S_h = tf.matmul(ht, self.W_h_a)  # [b, h]
        S_h = tf.expand_dims(S_h, 1)
        print('S_it:', S_h)  # [b, 1, h]
        S_emb = tf.reshape(tf.matmul(tf.reshape(repre, [-1, self.hidden_size]), self.W_emb_a),
                           [-1, self.num_neighbors, self.hidden_size])  # [b, N, h]
        print('S_emb:', S_emb)
        tanh = tf.tanh(S_h + S_emb)  # [b, N, h]
        print("tanh:", tanh)
        s = tf.reshape(tf.squeeze(tf.matmul(tf.reshape(tanh, [-1, self.hidden_size]), self.W_v_a)),
                       [-1, self.num_neighbors])  # [b, N]
        print("s:", s)  # [b, N]
        s_inf_mask = self.mask_softmax(mask, s)
        print(s_inf_mask) # [b, N]
        score = self.normalize_softmax(s_inf_mask)  # [b, N]
        print('score:', score)
        return score

    def _build_final_pred_A(self, PG_A, PS_A, s_pred_A, g_pred_A):
        with tf.variable_scope('final_predict_A'):
            pred_A = PG_A * g_pred_A + PS_A * s_pred_A
            print(pred_A)  # [b, num_items_A]
        return pred_A

    def _build_switch_B(self, encoder_state_B, transfer_state_A, graph_state, nei_mask):
        with tf.variable_scope('switch_B'):
            graph_rep = tf.reshape(graph_state, [-1, self.num_neighbors, self.hidden_size])
            nei_mask = tf.expand_dims(nei_mask, -1)
            graph_rep = nei_mask * graph_rep
            graph_rep = tf.reduce_sum(graph_rep, axis=1)
            concat_output = tf.concat([encoder_state_B[-1], transfer_state_A[-1], graph_rep], axis=-1)
            linear_switch = tf.layers.Dense(1, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            switch_matrix = linear_switch(concat_output)  # Tensor shape (b, 1)
            PG_B = tf.sigmoid(switch_matrix)
            PS_B = 1 - PG_B
        return PG_B, PS_B

    def _build_s_decoder_B(self, num_items_B, encoder_state_B, transfer_state_A, dropout_rate):
        with tf.variable_scope('s_predict_B'):
            concat_output = tf.concat([encoder_state_B[-1],transfer_state_A[-1]],axis=-1)
            concat_output = tf.nn.dropout(concat_output, rate=dropout_rate)
            pred_B = tf.layers.dense(concat_output, num_items_B,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            pred_B = tf.nn.softmax(pred_B)
            # pred_B = self.normalize_softmax(pred_B)
            print(pred_B) # [b, num_B]
        return pred_B

    def _build_g_decoder_B(self, ht, graph_state, num_items_B, nei_mask, nei_index_B, nei_is_in_B):
        with tf.variable_scope('g_predict_B'):
            self.W_h_b = random_weight(self.hidden_size, self.hidden_size, name='W_h_b')
            self.W_emb_b = random_weight(self.hidden_size, self.hidden_size, name='W_emb_b')
            self.W_v_b = random_weight(self.hidden_size, 1, name='W_v_b')
            graph_state = tf.reshape(graph_state, [-1, self.num_neighbors, self.hidden_size])  # [b, N, h]
            nei_mask = tf.expand_dims(nei_mask, -1)
            graph_state = nei_mask * graph_state
            att = self.g_decode_attention_B(ht[-1], graph_state, nei_is_in_B)  # [b, N]
            g_pred_B = tf.scatter_nd(nei_index_B, att, (tf.shape(graph_state)[0], num_items_B))
            print(g_pred_B)  # [b, num_item_B]
        return g_pred_B, att
    def g_decode_attention_B(self, ht, repre, mask):
        S_h = tf.matmul(ht, self.W_h_b)  # [b, h]
        S_h = tf.expand_dims(S_h, 1)
        print('S_it:', S_h)  # [b, 1, h]
        S_emb = tf.reshape(tf.matmul(tf.reshape(repre, [-1, self.hidden_size]), self.W_emb_b),
                           [-1, self.num_neighbors, self.hidden_size])  # [b, N, h]
        print('S_emb:', S_emb)
        tanh = tf.tanh(S_h + S_emb)  # [b, N, h]
        print("tanh:", tanh)
        s = tf.reshape(tf.squeeze(tf.matmul(tf.reshape(tanh, [-1, self.hidden_size]), self.W_v_b)),
                       [-1, self.num_neighbors])  # [b, N]
        print("s:", s)  # [b, N]
        s_inf_mask = self.mask_softmax(mask, s)
        print(s_inf_mask) # [b, N]
        score = self.normalize_softmax(s_inf_mask)  # [b, N]
        print('score:', score)
        return score

    def _build_final_pred_B(self, PG_B, PS_B, s_pred_B, g_pred_B):
        with tf.variable_scope('final_predict_B'):
            pred_B = PG_B * g_pred_B + PS_B * s_pred_B
            print(pred_B)
        return pred_B


    def _build_loss(self, ground_truth_A, pred_A, ground_truth_B, pred_B):
        loss_A = tf.keras.losses.categorical_crossentropy(tf.one_hot(ground_truth_A, depth=pred_A.shape[1]),pred_A)
        self.loss_A = tf.reduce_mean(loss_A, name='loss_A')
        
        loss_B = tf.keras.losses.categorical_crossentropy(tf.one_hot(ground_truth_B, depth=pred_B.shape[1]),pred_B)
        self.loss_B = tf.reduce_mean(loss_B, name='loss_B')
        
        loss = self.loss_A + self.loss_B
        
        loss_m_A = -(1 - tf.sign(self.ground_A_in_nei)) * tf.log(self.PS_A + 0.0001)
        self.loss_m_A = tf.reduce_mean(loss_m_A, name='loss_m_A')
        loss_m_B = -(1 - tf.sign(self.ground_B_in_nei)) * tf.log(self.PS_B + 0.0001)
        self.loss_m_B = tf.reduce_mean(loss_m_B, name='loss_m_B')
        loss_m = self.loss_m_A + self.loss_m_B
        loss_all = loss + loss_m

        return loss_all

    def _build_train_op(self, loss, lr):
        if self.optimizer_name == "Adam":            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)                                
        elif self.optimizer_name == "RMSProp":
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.9)
        elif self.optimizer_name == "AdaGrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
                  
        gradients = self.optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]

        train_op = self.optimizer.apply_gradients(capped_gradients)
        return train_op


def random_weight(dim_in, dim_out, name=None):
    return tf.get_variable(dtype=tf.float32, name=name, shape=[dim_in, dim_out], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

def random_bias(dim, name=None):
    return tf.get_variable(dtype=tf.float32, name=name, shape=[dim], initializer=tf.constant_initializer(1.0))

def matrix_mutliply(finstate, W,b,N,h):
    fin_state_new = tf.reshape(tf.matmul(finstate, W) + b, [-1, N, h])
    return fin_state_new




