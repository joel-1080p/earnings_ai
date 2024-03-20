# python=3.8
import numpy as np

from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt



######################
# HYPER PARAMETERS
######################
DATASET_FILE_PATH = 'dataset.csv'
EPOCH = 100
TEST_PERCENTAGE = .1
BATCH_SIZE = 2
NUMBER_OF_INPUTS = 3
VALIDATION_SPLIT = 0.2
######################
######################
######################



######################
# PREPARING DATA
######################
# Loads dataset from CSV to Numpy.
# Gets values as float.
dataset = np.loadtxt(DATASET_FILE_PATH, delimiter=",", dtype=float)

# Splits the inputs and targets.
inputs = dataset[:,0:NUMBER_OF_INPUTS]
targets_from_file = dataset[:,-1]

# Converts numerical targets to words.
targets = []
for target in targets_from_file:
    if target == 3:
        targets.append(2)
    if target == 2:
        targets.append(1)
    if target == 1:
        targets.append(0)
targets = np.array(targets)

# Splits the training and test sets based on a percentage given.
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                         targets,
                                                                         test_size = TEST_PERCENTAGE,
                                                                         random_state = 1)

# Shuffles both inputs and targets while keeping their relative positions.
inputs_train, targets_train = shuffle(inputs_train, targets_train)

# Normalizes the data from 1 to 0.
inputs_train_scaled = normalize(inputs_train, axis=1)
inputs_test_scaled = normalize(inputs_test, axis=1)


targets_train = to_categorical(targets_train, num_classes=3)
targets_test = to_categorical(targets_test, num_classes=3)

######################
# PREPARING MODEL
######################

#optimizer = Adam(lr=0.001)
optimizer = SGD(lr=0.001)

model = Sequential()

# Hidden Layers
model.add(Dense(units = 30, input_dim = NUMBER_OF_INPUTS, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 30, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 30, input_dim = NUMBER_OF_INPUTS, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 30, activation = 'relu'))
model.add(Dropout(0.2))


# Output layer
model.add(Dense(units = 3, activation = 'softmax'))
#model.add(Dense(units = 3, activation = 'softmax'))



######################
# TRAINING MODEL
######################

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


model.fit(x = inputs_train, y = targets_train, epochs = EPOCH, shuffle = True, batch_size = BATCH_SIZE)
#history = model.fit(x = inputs_train, y = targets_train, epochs = EPOCH, shuffle = True, batch_size = BATCH_SIZE, validation_split=VALIDATION_SPLIT)




######################
# VALIDATING TEST
######################
loss, accuracy = model.evaluate(inputs_test, targets_test)

print('\nEVALUATION')
print('Loss: %.2f' % (loss))
print('Accuracy: %.2f' % (accuracy*100))






######################
# PREDICTION
######################
predictions = model.predict(inputs_test)

print('\n\n')
show_num = 5
for i in range(len(predictions)):

    if i == show_num & show_num != 0:
        break

    print(f"PREDICTION SCALED: {predictions[i]}")
    print(f"PREDICTION : {np.argmax(predictions[i])}")
    print('\n')
    print(f"INPUTS: {inputs_test[i]}")
    print('###########################################')
    print('###########################################\n')



######################
# SAVE MODEL
######################
#model.save('earnings_ai.model')

#print(history.history.keys())
# loss, accuracy

# summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

del model


