import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils import shuffle


random.seed(1)
pd.set_option("display.max_column", None)
pd.set_option("display.max_colwidth", None)

data = pd.read_csv('data.csv')

data = data[['pokedex_number', 'name', 'status', 'height_m', 'weight_kg', 'total_points', 'hp', 'attack', 'defense',
             'sp_attack', 'sp_defense', 'speed']]


data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

X = data[['name', 'height_m', 'weight_kg', 'hp', 'attack', 'defense',
           'sp_attack', 'sp_defense', 'speed']]

Y = data[['name', 'status']]

count = {}
for index, row in Y.iterrows():
    if row['status'] in count:
        count[row['status']] += 1
    else:
        count[row['status']] = 1

# plt.pie(count.values(), labels=count.keys(), autopct='%1.1f%%', shadow=True)
# plt.show()

Y = Y.replace('Sub Legendary', 'Legendary')
Y = Y.replace('Mythical', 'Normal')

X_normal = []
X_legendary = []

Y_normal = []
Y_legendary = []

for index, row in Y.iterrows():
    if row['status'] == 'Legendary':
        X_legendary.append(X.iloc[index])
        Y_legendary.append(Y.iloc[index])
    else:
        X_normal.append(X.iloc[index])
        Y_normal.append(Y.iloc[index])

# Divide into train and test data

test_size = 0.33  # percentage of the data that is going to be used for testing
nr_legendary_test_samples = int(len(X_legendary) * test_size)
nr_normal_test_samples = int(len(X_normal) * test_size)

random_normal_samples = random.sample(X_normal, nr_normal_test_samples)
random_legendary_samples = random.sample(X_legendary, nr_legendary_test_samples)

normal_test_names = []
legendary_test_names = []
for sample in random_normal_samples:
    normal_test_names.append(sample['name'])

for sample in random_legendary_samples:
    legendary_test_names.append(sample['name'])

X_train = []
Y_train = []
X_test = []
Y_test = []

# append the normal pokemons
for pokemon in X_normal:
    values = np.delete(pokemon.values, 0)  # remove the name since it won't be used for training
    if pokemon.values[0] in normal_test_names:
        X_test.append(list(values))
    else:
        X_train.append(list(values))

# append the legendary pokemons
for pokemon in X_legendary:
    values = np.delete(pokemon.values, 0)  # remove the name since it won't be used for training
    if pokemon.values[0] in legendary_test_names:
        X_test.append(list(values))
    else:
        X_train.append(list(values))

# repeat the process for the Y set

# append the normal pokemons
for pokemon in Y_normal:
    values = np.delete(pokemon.values, 0)  # remove the name since it won't be used for training
    if pokemon.values[0] in normal_test_names:
        Y_test.append(0)
    else:
        Y_train.append(0)

# append the legendary pokemons
for pokemon in Y_legendary:
    values = np.delete(pokemon.values, 0)  # remove the name since it won't be used for training
    if pokemon.values[0] in legendary_test_names:
        Y_test.append(1)
    else:
        Y_train.append(1)

X_train, Y_train = shuffle(X_train, Y_train)

'''
# Saving training and testing files
np.savetxt("Processed Data\X_train.txt", X_train)
np.savetxt("Processed Data\X_test.txt", X_test)
np.savetxt("Processed Data\Y_train.txt", Y_train)
np.savetxt("Processed Data\Y_test.txt", Y_test)
'''
# -------------------------------------------------------------------------------- #

# Creating the neural network
'''
possible_weights = np.linspace(1, 4, 15)
possible_hidden_layers = [[5], [10], [20], [5, 4], [10, 5], [10, 10], [20, 6], [20, 4], [20, 10, 5]]
best_f1_score = -1.0

for weight in possible_weights:
    for config in possible_hidden_layers:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(len(X_train[0]),)))
        for nr_layer in range(len(config)):
            model.add(tf.keras.layers.Dense(config[nr_layer], activation='relu'))

        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        class_weights = {
            0: 1,
            1: weight
        }
        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs=200, class_weight=class_weights, verbose=0)

        predictions = model.predict(X_test)
        f1 = f1_score(Y_test, predictions.round())
        if f1 > best_f1_score:
            best_f1_score = f1
            best_weight = weight
            best_config = config
            model.save('best_model')

print('Best f1 score:', best_f1_score)
print('Best weight:', best_weight)
print("Best config:", best_config)
'''
best_model = tf.keras.models.load_model("best_model")
predictions = best_model.predict(X_test)
baseline = (len(Y_test) - sum(Y_test)) / len(Y_test)
# print('baseline:', baseline)
accuracy = accuracy_score(Y_test, predictions.round())
print('accuracy:', accuracy)
f1 = f1_score(Y_test, predictions.round())
print('f1 score:', f1)
conf_matrix = confusion_matrix(Y_test, predictions.round())
print('Confusion matrix:', conf_matrix)


def classify(name):
    stats = list(np.squeeze(X.loc[X['name'] == name].drop(columns=['name']).to_numpy()))
    _prediction = best_model.predict([stats]).round()
    return 'Legendary' if _prediction == 1 else 'Normal'


print('Classification of Latias: {0}\nTrue classification: {1}'.format(classify('Latias'), Y.loc[Y['name'] == 'Latias']['status'].values[0]))

