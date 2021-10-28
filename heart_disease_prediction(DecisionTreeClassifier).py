
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv("heart.csv")


"""heart_data.head()"""

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = DecisionTreeClassifier()

model.fit(X_train, Y_train)

Y_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train_prediction, Y_train)

print("Accuracy on training data: " , training_data_accuracy)

Y_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test_prediction, Y_test)

print("Accuracy on testing data: " , testing_data_accuracy)