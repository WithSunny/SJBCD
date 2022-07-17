import time
import _pickle as cPickle
from sjbcd import builde_model, read_file
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.metrics import classification_report

def evaluate(detection_data_file):
    max_opcodes_length = 300
    tokenizer_save_path = 'sjbcd/something_files/tokenizer-bcb.pkl'

    test_opcodes1, test_opcodes2, test_labels = read_file(detection_data_file)
    tokenizer = cPickle.load(open(tokenizer_save_path, 'rb'))
    test_indexs1 = tokenizer.texts_to_sequences(test_opcodes1)
    test_indexs2 = tokenizer.texts_to_sequences(test_opcodes2)
    test_indexs1 = pad_sequences(test_indexs1, maxlen=max_opcodes_length)
    test_indexs2 = pad_sequences(test_indexs2, maxlen=max_opcodes_length)
    embedding_matrix_path = 'sjbcd/something_files/embedding_glove200_matrix.npy'
    embedding_dim = 200
    model_save_path = 'sjbcd/something_files/sjbcd.bcb.h5'
    embedding_matrix = np.load(embedding_matrix_path, 'r')
    sum_opcodes = len(embedding_matrix)
    model = builde_model(embedding_matrix, sum_opcodes, embedding_dim, max_opcodes_length)
    model.load_weights(model_save_path)
    start_time = time.time()
    y_pred = model.predict([test_indexs1, test_indexs2], batch_size=100)
    yy_pred = np.rint(y_pred)
    print(classification_report(test_labels, yy_pred, digits=4))
    end_time = time.time()
    print("cost time：" + str(end_time - start_time) + "s")

test_file = 'sjbcd/dataset/CompiledBCB_test.csv'
evaluate(test_file)

