# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:34:23 2018

@author: Peter
"""

import tensorflow as tf
from bidaf import BiDAF
from prepro import load_embedding,sentence2index
import numpy as np
import jieba

def main():
    trained_model = "save/model.ckpt"
    embedding_file = "D:/DataMining/QASystem/wiki/wiki.zh.text.vector"
    embedding_size = 60 #word embedding维度
    trained_model = "save/model.ckpt"
    hidden_size = 100 #隐藏层神经元数量
    keep_prob = 1 #0.8
    
    question = jieba.lcut(input('问题：'))
    question_len = len(question)
    
    evidence = jieba.lcut(input('证据：'))
    evidence_len = len(evidence)
    embeddings,word2idx = load_embedding(embedding_file)
    questionIdx = sentence2index(question,word2idx,question_len)
    evidenceIdx = sentence2index(evidence,word2idx,evidence_len)
    with tf.Graph().as_default():
        with tf.variable_scope('Model'):
            model = BiDAF(embeddings,question_len,evidence_len,embedding_size,hidden_size,keep_prob)
            with tf.Session().as_default() as sess:
                saver = tf.train.Saver()
                print("开始加载模型")
                saver.restore(sess,trained_model)
                print("加载模型完毕")
                feed_dict = {
                    model.x: np.array([evidenceIdx]),
                    model.q: np.array([questionIdx])
                }
                p1,p2 = sess.run([model.logit_s,model.logit_e],feed_dict)
                p1 = np.argmax(p1[0])
                p2 = np.argmax(p2[0])
                print(p1,p2)
                print(evidence[p1],evidence[p2])

if __name__ == "__main__":
    main()