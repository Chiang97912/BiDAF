# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:45:25 2018

@author: Peter
"""

import tensorflow as tf
from bidaf import BiDAF
from prepro import get_max_length, load_embedding, load_data, next_batch

def main():
    # testing_file = "D:/DataMining/QASystem/new_data/test.ann.json"
    testing_file = "D:/DataMining/QASystem/new_data/validation.ann.json"
    # testing_file = "D:/DataMining/QASystem/new_data/training.json"
    trained_model = "save/model.ckpt"
    embedding_file = "D:/DataMining/QASystem/wiki/wiki.zh.text.vector"
    embedding_size = 60 #word embedding维度
    hidden_size = 100 #隐藏层神经元数量
    keep_prob = 1 #0.8
    batch_size = 60 #分批数据大小
    
    max_question_len,max_evidence_len = get_max_length(testing_file)
    embeddings,word2idx = load_embedding(embedding_file)
    questions,evidences,y1,y2 = load_data(testing_file,word2idx,max_question_len,max_evidence_len)
    with tf.Graph().as_default():
        with tf.variable_scope('Model'):
            model = BiDAF(embeddings,max_question_len,max_evidence_len,embedding_size,hidden_size,keep_prob)
            with tf.Session().as_default() as sess:
                saver = tf.train.Saver()
                print("开始加载模型")
                saver.restore(sess,trained_model)
                print("加载模型完毕")
                #sess.run(tf.global_variables_initializer()) 前面已经使用restore恢复变量了，如果再使用global_variables_initializer，会导致所有学习到的东西清零
                for batch_questions,batch_evidences,batch_y1,batch_y2 in next_batch(questions,evidences,y1,y2,batch_size):
                    feed_dict = {
                        model.x: batch_evidences,
                        model.q: batch_questions,
                        model.y1: batch_y1,
                        model.y2: batch_y2
                    }
                    acc_s,acc_e = sess.run([model.acc_s,model.acc_e],feed_dict)
                    print('ACC_S: %s\t\tACC_E: %s'%(acc_s,acc_e))

if __name__ == "__main__":
    main()