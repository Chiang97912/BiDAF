# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 19:57:38 2018

@author: Peter
"""

import tensorflow as tf
from bidaf import BiDAF
from prepro import get_max_length, load_embedding, load_data, next_batch

if __name__ == "__main__":
    # def main():
    training_file = "D:/DataMining/QASystem/new_data/training.json"
    trained_model = "checkpoints/model.ckpt"
    embedding_file = "D:/DataMining/QASystem/wiki/wiki.zh.text.vector"
    embedding_size = 60  # word embedding维度
    epochs = 30  # 20
    batch_size = 60  # 分批数据大小
    hidden_size = 100  # 隐藏层神经元数量
    keep_prob = 0.8  # 0.8
    learning_rate = 0.01  # 0.001
    lrdown_rate = 0.9  # 0.8
    gpu_mem_usage = 0.75
    gpu_device = "/gpu:0"
    cpu_device = "/cpu:0"

    max_quelen, max_evilen = get_max_length(training_file)
    embeddings, word2idx = load_embedding(embedding_file)
    questions, evidences, y1, y2 = load_data(
        training_file, word2idx, max_quelen, max_evilen)
    with tf.Graph().as_default(), tf.device(cpu_device):
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_usage)
        # session_conf = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        with tf.variable_scope('Model'):
            model = BiDAF(embeddings, max_quelen, max_evilen, embedding_size, hidden_size, keep_prob)
            with tf.Session().as_default() as sess:  # config=session_conf
                saver = tf.train.Saver()
                print("开始训练")
                sess.run(tf.global_variables_initializer())
                for i in range(epochs):
                    print("正在进行第%s次迭代训练" % (i+1))
                    for batch_questions, batch_evidences, batch_y1, batch_y2 in next_batch(questions, evidences, y1, y2, batch_size):
                        feed_dict = {
                            model.x: batch_evidences,
                            model.q: batch_questions,
                            model.y1: batch_y1,
                            model.y2: batch_y2,
                            model.lr: learning_rate
                        }
                        _, loss, acc_s, acc_e = sess.run(
                            [model.train, model.loss, model.acc_s, model.acc_e], feed_dict)
                        print('LOSS: %s\t\tACC_S: %s\t\tACC_E: %s' %
                              (loss, acc_s, acc_e))
                    learning_rate *= lrdown_rate
                    saver.save(sess, trained_model)
                print("训练结束")

# if __name__ == "__main__":
#    main()
