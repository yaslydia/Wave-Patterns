from scipy.io import loadmat
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np

Data = loadmat('./v5.mat')
testData = loadmat('./testing_v5.mat')

# output_array = Data['output_array']
parm_mat = Data['parm_mat']
Ex = Data['Wav_mat']
testEx = testData['Wav_mat']

Input = [Ex[i][0] for i in range(Ex.shape[0] - 1)]
Input = np.array(Input)
# Output = output_array[:, [0, 1]]
Output = parm_mat
n_sample = len(Input)  # number of samples
testInput = [testEx[i][0] for i in range(testEx.shape[0])]
testInput = np.array(testInput)

min_max_scaler = preprocessing.MinMaxScaler()  # normalization
# Input_scale = min_max_scaler.fit_transform(Input)
Output_scale = min_max_scaler.fit_transform(Output)
# testInput = min_max_scaler.fit_transform(testInput)

# visualize
# plt.imshow(Input[0])

index = np.arange(n_sample)
np.random.shuffle(index)
train_ratio = 0.7
x_train = Input[index[:int(0.7*n_sample)]]
x_test = Input[index[int(0.7*n_sample):]]
y_train = Output[index[:int(0.7*n_sample)]]
y_test = Output[index[int(0.7*n_sample):]]

# preprocess
img_x, img_y = 69, 129
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

# construct CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_x, img_y, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='relu'))

# compilation
model.compile(optimizer='adam', loss='mse')

# train
history = model.fit(x_train, y_train, batch_size=20, epochs=50)

# evaluate
score = model.evaluate(x_test, y_test)  # loss on test data

# plot loss
loss = history.history['training loss']
acc = history.history['training accuracy']
epochs = range(1, len(loss)+1)
plt.title('Loss')
plt.xlabel('epoch')
plt.yscale('log')
plt.ylabel('loss')
plt.plot(epochs, loss)
plt.show()
