from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
clf = BernoulliNB()
clf=LinearSVC(C=1e9)
clf=SVC(kernel="rbf")
