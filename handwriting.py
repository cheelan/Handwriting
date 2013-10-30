import read_data
import math
import pylab as pl
#See http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier


#Read dataset
train_images, train_labels = read_data.read(range(10), 'training')
test_images, test_labels = read_data.read(range(10), 'testing')

def svms(images, labels, cs, kernels):
  for k in kernels:
    for training_penalty in cs:
      clf = svm.SVC(kernel='linear', C=training_penalty)
      scores = cross_validation.cross_val_score(clf, images, labels, cv=5)
      print "Testing SVM kernel = " + k + " c = " + str(training_penalty) + " " + str(scores)

#The main parameters to adjust when using these methods is n_estimators and max_features. The former is the number of trees in the forest. The larger the better, but also the longer it will take to compute. In addition, note that results will stop getting significantly better beyond a critical number of trees. The latter is the size of the random subsets of features to consider when splitting a node. The lower the greater the reduction of variance, but also the greater the increase in bias. 
#See Feature importance evaluation: http://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
def random_forests(images, labels, estimators, features):
  for e in estimators:
    for f in features:
      clf = ExtraTreesClassifier() #num_estimators=e, max_features=f  
      scores = cross_validation.cross_val_score(clf, images, labels, cv=5)
      print "Testing random forests num_estimators = " + str(e) + " max features = " + str(f) + " " + str(scores)

  #Get the most important features (helpful visualization)
  #Tutorial: http://scikit-learn.org/0.13/auto_examples/ensemble/plot_forest_importances_faces.html#example-ensemble-plot-forest-importances-faces-py
  clf = ExtraTreesClassifier(compute_importances=True) #Use optimal parameters here
  clf.fit(images, labels)
  importances = clf.feature_importances_
  importances = importances.reshape(28,28)
  pl.matshow(importances, cmap=pl.cm.hot)
  pl.title("Pixel importances with forests of trees")
  pl.show()

svms(train_images[:10000], train_labels[:10000], [1], ['linear'])

