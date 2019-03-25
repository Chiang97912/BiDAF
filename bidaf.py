# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 20:20:44 2018

@author: Peter
"""

import tensorflow as tf


class BiDAF(object):
    def __init__(self, embeddings, quelen, conlen, embedding_size, hidden_size, keep_prob, stddev=0.1):
        self.embeddings = embeddings
        self.que_time_step = quelen
        self.con_time_step = conlen
        # self.embedding_size = embedding_size
        self.embedding_size = hidden_size
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.stddev = stddev

        self.x = tf.placeholder(tf.int32, [None, self.con_time_step])
        self.q = tf.placeholder(tf.int32, [None, self.que_time_step])
        self.y1 = tf.placeholder(tf.int32, [None, None])
        self.y2 = tf.placeholder(tf.int32, [None, None])
        self.lr = tf.placeholder(tf.float32)

        self.X, self.Q = self.embedding_layer(self.x, self.q, self.con_time_step, self.que_time_step, self.embeddings)

        self.H, self.U = self.contextual_embedding_layer(self.X, self.Q, self.con_time_step, self.que_time_step, self.hidden_size, self.embedding_size, self.keep_prob)
        self.G = self.attention_flow_layer(self.H, self.U)
        self.M = self.modeling_layer(self.G, self.con_time_step, self.que_time_step, self.hidden_size, self.embedding_size, self.keep_prob)
        self.logit_s, self.logit_e = self.output_layer(self.G, self.M, self.con_time_step, self.hidden_size, self.embedding_size, self.keep_prob)

        self.train, self.loss = self.optimize(self.logit_s, self.logit_e, self.y1, self.y2, self.lr)
        self.acc_s = self.compute_accuracy(self.logit_s, self.y1)
        self.acc_e = self.compute_accuracy(self.logit_e, self.y2)

    def bidirectional_LSTM(self, X, time_step, hidden_size, embedding_size, keep_prob, output_dim=2, num_layers=1):
        inputs = X
        for _ in range(num_layers):
            with tf.variable_scope(None, default_name="bidirectional-rnn"):
                lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
                lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
                fw_drop = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
                bw_drop = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_drop, bw_drop, inputs, dtype=tf.float32)
                concat_outputs = tf.concat(outputs, 2)
                inputs = concat_outputs
        return concat_outputs

        '''
        W = tf.Variable(tf.random_normal([2*hidden_size,embedding_size*output_dim],stddev=self.stddev))
        b = tf.Variable(tf.constant(0.0,shape=[embedding_size*output_dim]))

        reshaped_output = tf.reshape(concat_outputs,[-1,2*hidden_size]) #(batch_size*time_step,2*hidden_size)
        reshaped_output = tf.matmul(reshaped_output,W) + b

        Y = tf.reshape(reshaped_output,[-1,time_step,embedding_size*output_dim])
        #(batch_size,time_step,output_dim*d)
        return Y
        '''

    def embedding_layer(self, x, q, con_time_step, que_time_step, embeddings):
        embeddings = tf.constant(embeddings, dtype=tf.float32)
        embed_x = tf.nn.embedding_lookup(embeddings, x)
        embed_q = tf.nn.embedding_lookup(embeddings, q)

        return embed_x, embed_q

    def higyway_network(self, inp, size, layer_size=1):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """
        out = inp
        with tf.variable_scope('highway', reuse=tf.reuse):
            for idx in range(layer_size):
                out = tf.tanh(tf.layers.dense(inp, size, scope='dense1'))
                transform_gate = tf.sigmoid(tf.layers.dense(inp, size, scope='dense2'))
                carry_gate = 1. - transform_gate
                out = transform_gate * out + carry_gate * inp
        return out

    def contextual_embedding_layer(self, X, Q, con_time_step, que_time_step, hidden_size, embedding_size, keep_prob):
        with tf.variable_scope('Contextual_embeding_layer'):
            with tf.variable_scope('context'):
                H = self.bidirectional_LSTM(X, con_time_step, hidden_size, embedding_size, keep_prob, 2)
            with tf.variable_scope('question'):
                U = self.bidirectional_LSTM(Q, que_time_step, hidden_size, embedding_size, keep_prob, 2)
        return H, U

    def attention_flow_layer(self, H, U):
        with tf.variable_scope('Attention_flow_layer'):
            # B = H.get_shape().as_list()[0] # batch_size
            T = H.get_shape().as_list()[1]  # length of context
            J = U.get_shape().as_list()[1]  # length of question
            D = H.get_shape().as_list()[-1]  # dimension

            H_trans = tf.transpose(H, [0, 2, 1])  # (B,2d,T)
            U_trans = tf.transpose(U, [0, 2, 1])  # (B,2d,J)

            # H:(B,T,D)
            HH = tf.expand_dims(H, 2)  # (B,T,1,D)
            HH = tf.tile(HH, [1, 1, J, 1])  # (B,T,J,D)
            # U:(B,J,D)
            UU = tf.expand_dims(U, 1)  # (B,1,J,D)
            UU = tf.tile(UU, [1, T, 1, 1])

            H_mul_U = tf.multiply(HH, UU)  # (B,T,J,D)
            H_mul_U = tf.reshape(H_mul_U, [-1, T * J, D])  # (B,T*J,D)

            HHH = tf.reshape(HH, [-1, T * J, D])  # (B,T*J,D)
            # UUU = tf.tile(U,[1,T,1]) # (B,T*J,D)
            UUU = tf.reshape(UU, [-1, T * J, D])

            concat = tf.concat([HHH, UUU, H_mul_U], 2)  # (B,T*J,3D) 不换行并列连接

            w_s = tf.Variable(tf.random_normal([3 * D, 1], stddev=self.stddev))

            reshape = tf.reshape(concat, [-1, 3 * D])  # (B*T*J,3*D)
            alpha = tf.matmul(reshape, w_s)  # (B*T*J,1)

            S = tf.reshape(alpha, [-1, T, J])  # (B,T,J)

            a = tf.nn.softmax(S, -1)
            a_trans = tf.transpose(a, [0, 2, 1])  # (B,J,T)
            attended_U = tf.matmul(U_trans, a_trans)  # (B,2d,T)

            # 先在column上使用maximum function，再softmax
            b = tf.nn.softmax(tf.reduce_max(S, 2), -1)
            # (B,T) column上的维度被压缩了
            b = tf.expand_dims(b, -1)  # (B,T,1)
            attended_H = tf.matmul(H_trans, b)  # (B,2d,1)
            attended_H = tf.tile(attended_H, [1, 1, T])

            h_el_mul_att_u = tf.multiply(H_trans, attended_U)  # (B,2d,T)
            h_el_mul_att_h = tf.multiply(H_trans, attended_H)  # (B,2d,T)

            G = tf.concat([H_trans, attended_U, h_el_mul_att_u,
                           h_el_mul_att_h], axis=1)  # (B,8d,T)

            return G

    def modeling_layer(self, G, con_time_step, que_time_step, hidden_size, embedding_size, keep_prob):
        with tf.variable_scope('Modeling_layer'):
            # G = tf.unstack(G,con_time_step,axis=2) # ?
            G = tf.transpose(G, [0, 2, 1])
            M = self.bidirectional_LSTM(G, con_time_step, hidden_size, embedding_size, keep_prob, 2, num_layers=2)
            # (?,con_time_step,2d)
            M = tf.transpose(M, [0, 2, 1])
            # (?,2d,con_time_step)

            return M

    def output_layer(self, G, M, con_time_step, hidden_size, embedding_size, keep_prob):
        with tf.variable_scope('Output_layer'):
            w_p1 = tf.Variable(tf.random_normal([10 * embedding_size, 1], stddev=self.stddev))  # (10d,1)
            G_M = tf.concat([G, M], axis=1)  # (?,10d,con_time_step)
            G_M = tf.transpose(G_M, [0, 2, 1])  # (?,con_time_step,10d)
            G_M = tf.reshape(G_M, [-1, 10 * embedding_size])
            p1 = tf.matmul(G_M, w_p1)  # (?*con_time_step,1)
            # (?,con_time_step)
            p1 = tf.reshape(p1, [-1, M.get_shape().as_list()[-1]])

            # M = tf.unstack(M,con_time_step,axis=2)
            M2 = tf.transpose(M, [0, 2, 1])
            M2 = self.bidirectional_LSTM(M2, con_time_step, hidden_size, embedding_size, keep_prob, 2)  # (?,con_time_step,2d)
            M2 = tf.transpose(M2, [0, 2, 1])
            G_M2 = tf.concat([G, M2], axis=1)

            # M = tf.stack(M,axis=2)
            w_p2 = tf.Variable(tf.random_normal([10 * embedding_size, 1], stddev=self.stddev))  # (10d,1)
            G_M2 = tf.transpose(G_M2, [0, 2, 1])
            G_M2 = tf.reshape(G_M2, [-1, 10 * embedding_size])
            p2 = tf.matmul(G_M2, w_p2)
            # (?,con_time_step)
            p2 = tf.reshape(p2, [-1, M.get_shape().as_list()[-1]])

            return p1, p2

    def optimize(self, logit_s, logit_e, y1, y2, lr):
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_s, labels=y1))
        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_e, labels=y2))
        loss = loss1 + loss2
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train = optimizer.apply_gradients(zip(grads, tvars))

        return train, loss

    def compute_accuracy(self, logits, labels):
        correct_prediction = tf.equal(
            tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy
