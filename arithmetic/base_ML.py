from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
sklearn_ML={
    'BernoulliNB':BernoulliNB(),
    'LinearSVC':LinearSVC(C=1e9),
}