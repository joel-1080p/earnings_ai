# python=3.8
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
import pickle
import warnings

######################
# HYPER PARAMETERS
######################
MAX_ITER = 1000
TRAIN_PERCENTAGE = 0.75
BATCH_SIZE = 75
SOLVER = 'adam'
ACTIVATION = 'relu'
LEARNING_RATE = 'constant'
LEARNING_RATE_INIT = 0.001
HIDDEN_LAYERS = (7, 33, 32, 3)
K_FOLDS = 10
VERBOSE = 0
SHUFFLE = True
######################
######################
######################


######################
# HYPER PARAMETERS
######################
DATASET_FILE_PATH = 'dataset.csv'
SAVE_HYPER_PARAMETERS = True
PRINT_METRICS = True
SAVE_MODEL = False
warnings.filterwarnings("ignore")
######################
######################
######################


######################
# PREPARING DATA
######################
# Loads dataset from CSV to Pandas.
df_model = pd.read_csv(DATASET_FILE_PATH, header=None)

# Gets the raw values and places into a dataframe.
df_model = df_model.rename(columns={0: 'pe', 
                                  1: 'price_book',
                                  2: 'roe', 
                                  3: 'roa',
                                  4: 'debt_to_equity', 
                                  5: 'gross_margin',
                                  6: 'operating_margin', 
                                  7: 'current_ratio',
                                  8: 'quick_ratio',
                                  9: 'price_fcf',
                                  10: 'eps', 
                                  11: 'book_value_per_share',
                                  12: 'interest_coverage',
                                  13: 'asset_turnover',
                                  14: 'debt_asset',
                                  15: 'target'})

# X Inputs.
attributes = ['pe',
              'price_book',
              'roe',
              'roa',
              'debt_to_equity',
              'gross_margin',
              'operating_margin',
              'current_ratio',
              'quick_ratio',
              'price_fcf',
              'eps',
              'book_value_per_share',
              'interest_coverage',
              'asset_turnover',
              'debt_asset']

# Creates inputs and targets.
X = df_model[attributes]
y = df_model['target']

# Allocates the index for testing (validation).
train_pct_index = int(TRAIN_PERCENTAGE * len(X))

# Creates training sets and testing sets.
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]

# Normalize inputs.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


######################
# TRAINING
######################
temp_acc = 0
temp_pos = 0
for i in range(1):

    # Creates neural network.
    NN = MLPClassifier(hidden_layer_sizes = HIDDEN_LAYERS,
                    random_state = 100,
                    verbose = VERBOSE,
                    max_iter = MAX_ITER,
                    activation = ACTIVATION,
                    solver = SOLVER,
                    batch_size = BATCH_SIZE,
                    learning_rate = LEARNING_RATE,
                    shuffle = SHUFFLE,
                    n_iter_no_change = MAX_ITER)

    # Trains network.
    NN.fit(X_train, y_train)

    print("RESULTS")
    print("Cross Val Score Recall")
    cross_val_score_recall = np.mean(cross_val_score(NN, X_train,y_train, cv=K_FOLDS, scoring='precision'))
    print(cross_val_score_recall)


    ######################
    # PREDICTING
    ######################
    pred_train = NN.predict(X_train)
    pred_test = NN.predict(X_test)
    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)

    if cross_val_score_recall > temp_acc:
        temp_acc = cross_val_score_recall
        temp_pos = i

    print(f'pos {i} done at {cross_val_score_recall}')
    print(f'best so far : {temp_acc} at {temp_pos}')

    print("="*20)
    print('****Train Results****')
    print("Accuracy: {:.4%}".format(acc_train))
    print('****Test Results****')
    print("Accuracy: {:.4%}".format(acc_test)) 
    print('\n')

print(f'temp_acc : {temp_acc}')
print(f'temp_pos : {temp_pos}')


######################
# SAVE MODEL
######################
if SAVE_MODEL:
    with open('model.pkl','wb') as f:
        pickle.dump(NN,f)



######################
# METRICS
######################
if PRINT_METRICS:
    print("VALUE COUNTS")
    print(df_model['target'].value_counts())
    print('\n\n')

    print('CONFUSION MATRIX')
    print('matrix_train')
    matrix_train = confusion_matrix(y_train, pred_train)
    print(matrix_train)
    print('\n')

    print('matrix_test')
    matrix_test = confusion_matrix(y_test, pred_test)
    print(matrix_test)
    print('\n')

    print('CLASSIFICATION REPORT')
    print('report_train')
    report_train = classification_report(y_train, pred_train)
    print(report_train)
    print('\n')

    print('report_test')
    report_test = classification_report(y_test, pred_test)
    print(report_test)
    print('\n')

    print('PERMUTATION IMPORTANCE')
    results = permutation_importance(NN, X, y, scoring='neg_mean_squared_error')
    # get importance
    importance = results.importances_mean
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance

    pyplot.figure(num='Permutation Importance')
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

if SAVE_HYPER_PARAMETERS:
    with open('hyper_parameters.txt','a+') as f:
        f.write(f'train_acc :\t\t {acc_train} \n')
        f.write(f'test_acc :\t\t {acc_test} \n')
        f.write(f'hidden_layers :\t\t {HIDDEN_LAYERS} \n')
        f.write(f'max_iter :\t\t {MAX_ITER} \n')
        f.write(f'train_percentage :\t {TRAIN_PERCENTAGE} \n')
        f.write(f'batch_size :\t\t {BATCH_SIZE} \n')
        f.write(f'solver :\t\t {SOLVER} \n')
        f.write(f'activation :\t\t {ACTIVATION} \n')
        f.write(f'learning_rate :\t\t {LEARNING_RATE} \n')
        f.write(f'learning_rate_init :\t {LEARNING_RATE_INIT} \n')
        f.write(f'shuffle :\t\t {SHUFFLE} \n\n')
        f.write(f'data_split :\n{y.value_counts()} \n\n')
        f.write(f'matrix_train :\n {matrix_train} \n\n')
        f.write(f'matrix_test :\n {matrix_test} \n\n')
        f.write(f'report_train :\n {report_train} \n\n')
        f.write(f'report_test :\n {report_test} \n\n')
        f.write(f'########################################## \n')
        f.write(f'########################################## \n')
        f.write(f'########################################## \n\n\n')