import pandas as pd
from collections import Counter
from common import *
import getsplit
from cnn_model import *
from lstm_model import *
import accuracy
from sklearn.model_selection import train_test_split
import sys
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.metrics import classification_report

CSV_DIR = '../data/bio_metadata.csv'
EPOCHS = 2
DEBUG = True

# cf = np.asarray([[16,  5,  2],[ 3, 15,  3],[ 2,  2, 16]])
# show_confusion_matrix(cf, plt, ['mandarin', 'arabic', 'english'], 'cnn')



if __name__ == '__main__':
    '''
        Console command example:
        python main.py cnn
    '''


    # Load arguments
    # file_name = sys.argv[1]
    file_name = CSV_DIR
    network = sys.argv[1]

    if len(sys.argv) == 3:
        EPOCHS = int(sys.argv[2])
    else:
        EPOCHS = EPOCHS

    # Load metadata
    df = pd.read_csv(file_name)

    # Filter metadata to retrieve only files desired
    filtered_df = getsplit.filter_df(df)

    # Train test split
    X_train, X_test, y_train, y_test = getsplit.split_people(filtered_df)
    # print type(y_test)
    # Get statistics
    train_count = Counter(y_train)
    test_count = Counter(y_test)
    print train_count
    print test_count
    acc_to_beat = test_count.most_common(1)[0][1] / float(np.sum(test_count.values()))
    print acc_to_beat

    # To categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Get resampled wav files using multiprocessing
    if DEBUG:
        print('loading wav files')
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool = ThreadPool(4)
    print multiprocessing.cpu_count()
    X_train = pool.map(get_wav, X_train)
    X_test = pool.map(get_wav, X_test)

    # Convert to MFCC
    if DEBUG:
        print('converting to mfcc')
    X_train = pool.map(to_mfcc, X_train)
    X_test = pool.map(to_mfcc, X_test)

    # Create segments from MFCCs
    X_train, y_train = make_segments(X_train, y_train)
    # X_validation, y_validation = make_segments(X_test, y_test)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.15)
    # print "Validation shape: {}".format(X_validation)
    # Randomize training segments
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0)

    if network == 'cnn':
        # Train model
        model = train_model(np.array(X_train), np.array(y_train), np.array(X_validation),np.array(y_validation), EPOCHS)
        # Make predictions on full X_test MFCCs
        y_predicted = accuracy.predict_class_all(create_segmented_mfccs(X_test), model, 'cnn')
        class_sum = np.sum(accuracy.confusion_matrix(y_predicted, y_test),axis=1)
        confusion_matrix = accuracy.confusion_matrix(y_predicted, y_test)
        print confusion_matrix
        print accuracy.get_accuracy(y_predicted,y_test)
        show_confusion_matrix(confusion_matrix, plt, ['mandarin', 'arabic', 'english'], 'cnn')

    if network == 'lstm':
        # Train Lstm Model
        lstm = train_lstm_model(np.array(X_train), np.array(y_train), np.array(X_validation), np.array(y_validation), EPOCHS)
        y_predicted_lstm = accuracy.predict_class_all(create_segmented_mfccs(X_test), lstm, 'lstm')
        print np.sum(accuracy.confusion_matrix(y_predicted_lstm, y_test), axis=1)
        confusion_matrix = accuracy.confusion_matrix(y_predicted_lstm, y_test)
        print confusion_matrix
        # print(classification_report(y_test, y_predicted_lstm))
        print accuracy.get_accuracy(y_predicted_lstm, y_test)
        show_confusion_matrix(confusion_matrix, plt, ['mandarin', 'arabic', 'english'], 'lstm')

