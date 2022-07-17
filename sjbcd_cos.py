import codecs
from gensim.models import KeyedVectors as Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout, GRU
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import *
import _pickle
import matplotlib.pyplot as plt
from keras.layers.core import Lambda


def read_file(csv_file):
    opcodes1 = []
    opcodes2 = []
    labels = []
    with codecs.open(csv_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for values in reader:
            opcodes1.append(values[1])
            opcodes2.append(values[2])
            labels.append(int(values[3]))
    return opcodes1, opcodes2, labels

def init_input(train_file, test_file):
    train_opcodes1, train_opcodes2, train_labels = read_file(train_file)
    test_opcodes1, test_opcodes2, test_labels = read_file(test_file)
    return train_opcodes1, train_opcodes2, train_labels, test_opcodes1, test_opcodes2, test_labels

def get_tokenizer(train_opcodes1, train_opcodes2, test_opcodes1, test_opcodes2, tokenizer_save_path):
    tokenizer = Tokenizer(num_words=256, lower=False)
    tokenizer.fit_on_texts(train_opcodes1 + train_opcodes2 + test_opcodes1 + test_opcodes2)
    _pickle.dump(tokenizer, open(tokenizer_save_path, "wb"))
    return tokenizer


def get_indexs(train_opcodes1, train_opcodes2, test_opcodes1, test_opcodes2, tokenizer_save_path, max_opcodes_length):
    tokenizer = get_tokenizer(train_opcodes1, train_opcodes2, test_opcodes1, test_opcodes2, tokenizer_save_path)
    train_indexs1 = tokenizer.texts_to_sequences(train_opcodes1)
    train_indexs2 = tokenizer.texts_to_sequences(train_opcodes2)
    test_indexs1 = tokenizer.texts_to_sequences(test_opcodes1)
    test_indexs2 = tokenizer.texts_to_sequences(test_opcodes2)

    opcodes_directory = tokenizer.word_index
    train_indexs1 = pad_sequences(train_indexs1, maxlen=max_opcodes_length)
    train_indexs2 = pad_sequences(train_indexs2, maxlen=max_opcodes_length)
    test_indexs1 = pad_sequences(test_indexs1, maxlen=max_opcodes_length)
    test_indexs2 = pad_sequences(test_indexs2, maxlen=max_opcodes_length)
    return train_indexs1, train_indexs2, test_indexs1, test_indexs2, opcodes_directory

def get_embedding_matrix(opcodes_directory, embedding_matrix_path, embedding_model, embedding_dim):
    word2vec = Word2Vec.load(embedding_model)
    sum_opcodes =len(opcodes_directory) + 1
    embedding_matrix = np.zeros((sum_opcodes, embedding_dim))
    for word, i in opcodes_directory.items():
        if word in word2vec.index_to_key:
            embedding_matrix[i] = word2vec.word_vec(word)
    np.save(embedding_matrix_path, embedding_matrix)
    return embedding_matrix, sum_opcodes
def cosine(x1, x2):
    def _cosine(x):
        dot1 = K.batch_dot(x[0], x[1], axes=1)
        dot2 = K.batch_dot(x[0], x[0], axes=1)
        dot3 = K.batch_dot(x[1], x[1], axes=1)
        max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
        return dot1 / max_
    output_shape = (1,)
    value = Lambda(_cosine, output_shape=output_shape)([x1, x2])
    return value

def builde_model(embedding_matrix, sum_opcodes, embedding_dim, max_opcodes_length):
    embedding_layer = Embedding(sum_opcodes, embedding_dim, weights=[embedding_matrix], input_length=max_opcodes_length, trainable=False)
    gru_layer = GRU(units=175, dropout=0.15, recurrent_dropout=0.15)
    input1 = Input(shape=(max_opcodes_length,), dtype='int32')
    vectors1 = embedding_layer(input1)
    x1 = gru_layer(vectors1)
    input2 = Input(shape=(max_opcodes_length,), dtype='int32')
    vectors2 = embedding_layer(input2)
    x2 = gru_layer(vectors2)
    layer_data = cosine(x1, x2)
    layer_data = BatchNormalization()(layer_data)
    layer_data = Dropout(rate=0.15)(layer_data)
    layer_data = Dense(units=100, activation='relu')(layer_data)
    layer_data = BatchNormalization()(layer_data)
    layer_data = Dropout(0.15)(layer_data)
    prob = Dense(1, activation='sigmoid')(layer_data)
    model = Model(inputs=[input1, input2], outputs=prob)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
    model.summary()
    return model

def draw(history):
    acc = history.history['acc']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)
    plt.title('Accuracy and Loss')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, loss, 'blue', label='Validation loss')
    plt.legend()
    plt.show()

def train(train_indexs1, train_indexs2, train_labels, model, model_save_path, epochs_num, batch_size, isDraw=True):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=True)
    hist = model.fit([train_indexs1, train_indexs2], train_labels, validation_data=([train_indexs1, train_indexs2], train_labels),
                     epochs=epochs_num, batch_size=batch_size, shuffle=True,
                     callbacks=[early_stopping, model_checkpoint])
    model.load_weights(model_save_path)
    min_loss = min(hist.history['loss'])
    max_acc = max(hist.history['acc'])
    print(min_loss, max_acc)
    if isDraw:
        draw(hist)

if __name__ == '__main__':
    train_file = 'sjbcd/dataset/CompiledBCB_train.csv'
    test_file ='sjbcd/dataset/CompiledBCB_test.csv'
    max_opcodes_length = 300
    tokenizer_save_path = 'sjbcd/something_files/tokenizer-bcb.pkl'
    embedding_model_file = 'sjbcd/something_files/opcode2v_glove200.mod'
    embedding_matrix_path = 'sjbcd/something_files/embedding_glove200_matrix.npy'
    embedding_dim = 200
    model_save_path = 'sjbcd/something_files/sjbcd-cos_glove200.h5'
    epochs = 15
    batch_size = 1000

    train_opcodes1, train_opcodes2, train_labels, test_opcodes1, test_opcodes2, test_labels = init_input(train_file, test_file)
    train_indexs1, train_indexs2, test_indexs1, test_indexs2, opcodes_directory \
        = get_indexs(train_opcodes1, train_opcodes2, test_opcodes1, test_opcodes2, tokenizer_save_path, max_opcodes_length)
    embedding_matrix, sum_opcodes = get_embedding_matrix(opcodes_directory, embedding_matrix_path, embedding_model_file, embedding_dim)
    model = builde_model(embedding_matrix, sum_opcodes, embedding_dim, max_opcodes_length)
    train_labels = np.array(train_labels)
    train(train_indexs1, train_indexs2, train_labels, model, model_save_path, epochs, batch_size)


