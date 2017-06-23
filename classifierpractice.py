#gender classifier give height, weight, and shoe size

#tree
from sklearn import tree

clf = tree.DecisionTreeClassifier()

#[height, weight, shoe size]
x = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

clf = clf.fit(x, y)
prediction = clf.predict([[190, 70, 43]])
print('Decision Tree: ', prediction)


#KNeighbors
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(3)
clf = clf.fit(x, y)
prediction = clf.predict([[190, 70, 43]])
print('KNeighbors: ', prediction)


#LogisticRegresssion
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf = clf.fit(x, y)
prediction = clf.predict([[190, 70, 43]])
print('Logistic Regression: ', prediction)


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(x, y)
prediction = clf.predict([[190, 70, 43]])
print('Naive Bayes: ', prediction)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(2)
clf = clf.fit(x, y)
prediction = clf.predict([[190, 70, 43]])
print('Random Forest: ', prediction)