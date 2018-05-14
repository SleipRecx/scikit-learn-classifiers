from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np

data = [[0,0,0],[0,0,0],[0,0,0],[1,0,0],[1,0,1],[1,1,1],[1,1,1],[1,1,0],[0,1,1],[0,0,1]]
X = np.array(data)
Y = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])


test = np.array([[0,1,0]])

clf = BernoulliNB(alpha=0.000001)
#clf.fit(X, Y)
#print("Naive Bayes:", clf.predict_proba(test))


clf2 = DecisionTreeClassifier()
#clf2.fit(X, Y)
#print("DT:", clf2.predict_proba(test))

clf3 = KNeighborsClassifier()
#clf3.fit(X, Y)
#print("KNN", clf3.predict_proba(test))


clf4 = MLPClassifier(max_iter=1000)
#clf4.fit(X, Y)
#print("MLP", clf4.predict_proba(test))

clf5 = VotingClassifier(("NB", clf), ("DT", clf2), ("KNN", clf3), ("MLP", clf4))

for c, label in zip([clf, clf2, clf3, clf4, clf5], ['Naive Bayes', 'Decision Tree', 'KNN', 'MLP', "Ensemble"]):
    scores = cross_val_score(c, X, Y, cv=5, scoring='accuracy')
    print("Accuracy:", scores.mean(), scores.std(), label)


