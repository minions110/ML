from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import SpectralClustering
from sklearn .cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
sklearn_ML={
    #classify
    'BernoulliNB':BernoulliNB(),
    'LinearSVC':LinearSVC(C=1e9),
    'Adaboost':AdaBoostClassifier(n_estimators=100, random_state=0),
    #cluster
    'KMeans':KMeans(n_clusters=2),
    'specluster':SpectralClustering(gamma=0.1),
    'DBSCAN':DBSCAN(),
    'Birch':Birch()
}