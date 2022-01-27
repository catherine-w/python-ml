import tensorflow
from tensorflow import keras

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
# print(data.head())

data = data[["G1", "G2", "G3", "age", "studytime", "failures", "freetime", "goout", "Dalc", "Walc",
             "health", "absences", "higher", "activities", "paid", "schoolsup", "famsup"]]

# convert data to ints
data.higher = data.higher.replace({'yes': 1, 'no': 0})
data.activities = data.activities.replace({'yes': 1, 'no': 0})
data.paid = data.paid.replace({'yes': 1, 'no': 0})
data.schoolsup = data.schoolsup.replace({'yes': 1, 'no': 0})
data.famsup = data.famsup.replace({'yes': 1, 'no': 0})
print(data.head())

# we want to predict label based on all the diff attributes
# will remove final label G3 from list of attributes when testing data set

predict = "G3"

x = np.array(data.drop(columns=predict))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)


# best = 0
# for _ in range(24):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#     print(accuracy)
#
#     if accuracy > best:
#         best = accuracy
#         with open("studentmodel.pickle", "wb") as f:
#             pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Plotting data
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()