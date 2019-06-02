from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras import optimizers

LOG_DIR = '../logs/lstm/'


def train_lstm_model(X_train, y_train, X_validation, y_validation, EPOCHS, batch_size=128):
    # Get row, column, and class sizes
    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]
    val_rows = X_validation[0].shape[0]
    val_cols = X_validation[0].shape[1]
    num_classes = len(y_train[0])

    input_shape = (rows, cols)
    X_train = X_train.reshape(X_train.shape[0], rows, cols)
    X_validation = X_validation.reshape(X_validation.shape[0], val_rows, val_cols)

    lstm = Sequential()
    lstm.add(LSTM(64, return_sequences=True, stateful=False, input_shape=input_shape, activation='tanh'))
    lstm.add(LSTM(64, return_sequences=True, stateful=False, activation='tanh'))
    lstm.add(LSTM(64, stateful=False, activation='tanh'))

    # add dropout to control for overfitting
    lstm.add(Dropout(.25))

    # squash output onto number of classes in probability space
    lstm.add(Dense(num_classes, activation='softmax'))
    adam = optimizers.adam(lr=0.0001)
    lstm.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

    es = EarlyStopping(monitor='acc', min_delta=.005, patience=10, verbose=1, mode='auto')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir=LOG_DIR, histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)
    lstm.fit(X_train, y_train, batch_size=batch_size,
              epochs=EPOCHS, validation_data=(X_validation,y_validation),
              callbacks=[es,tb])
    return lstm
