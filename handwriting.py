import read_data
import math
import cProfile
import pylab as pl
import otsu
import numpy
#See http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html
from sklearn import cross_validation
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier


#Read dataset
train_images_gray, train_labels = read_data.read(range(10), 'training')
test_images_gray, test_labels = read_data.read(range(10), 'testing')
print "Done reading data"

def convert_bw(images):
  bw = []
  for i in images:
    bw.append(numpy.array(otsu.otsu(i)))

train_images_bw = convert_bw(train_images_gray)
test_images_bw = convert_bw(test_images_gray)



def svms(images, labels, test_images, test_labels, cs):
  for training_penalty in cs:
    clf = svm.SVC(C=training_penalty)
    #scores = cross_validation.cross_val_score(clf, images, labels, cv=5)
    scores = clf.fit(images, labels).score(test_images, test_labels)
    print "Testing SVM c = " + str(training_penalty) + " " + str(scores)
    #print "Testing SVM kernel = " + k + " c = " + str(training_penalty) + " " + str(scores)

#Doesn't necessarily use all of max_depth. TDIDT stops once all training examples are classified correctly
def decision_trees(images, labels, test_images, test_labels, depths, showimage=False):
    for d in depths:
      print d
      clf = DecisionTreeClassifier(max_depth = d)
      scores = clf.fit(images, labels).score(test_images, test_labels)
      print "Testing decision trees max depth = " + str(d) + " " + str(scores)

#The main parameters to adjust when using these methods is n_estimators and max_features. The former is the number of trees in the forest. The larger the better, but also the longer it will take to compute. In addition, note that results will stop getting significantly better beyond a critical number of trees. The latter is the size of the random subsets of features to consider when splitting a node. The lower the greater the reduction of variance, but also the greater the increase in bias. 
#See Feature importance evaluation: http://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
def random_forests(images, labels, test_images, test_labels, estimators, depths, showimage=False):
  for e in estimators:
    for d in depths:
      clf = ExtraTreesClassifier(n_estimators=e, max_depth=d,compute_importances=showimage) #num_estimators=e, max_features=f  
      #scores = cross_validation.cross_val_score(clf, images, labels, cv=5)
      scores = clf.fit(images, labels).score(test_images, test_labels)
      print "Testing random forests num_estimators = " + str(e) + " max depth = " + str(d) + " " + str(scores)
      if showimage:
        importances = clf.feature_importances_
        importances = importances.reshape(28,28)
        pl.matshow(importances, cmap=pl.cm.hot)
        pl.title("Pixel importances with forests of trees")
        pl.show()


#cProfile.run("svms(train_images[:10000], train_labels[:10000], [1], ['poly'])")
# print "BLACK WHITE"
# svms(train_images_bw, train_labels, test_images_bw, test_labels, [0.01, 0.1, 1, 10, 100])
# random_forests(train_images_bw, train_labels, test_images_bw, test_labels, [1, 10, 50, 100], [2, 4, 8, 16, 32], False)
# decision_trees(train_images_bw, train_labels, test_images_bw, test_labels, [2, 4, 8, 16, 32, 64, 128])

print "GRAYSCALE"
svms(train_images_gray, train_labels, test_images_gray, test_labels, [0.01, 0.1, 1, 10, 100])
random_forests(train_images_gray, train_labels, test_images_gray, test_labels, [1, 10, 50, 100], [2, 4, 8, 16, 32], False)
decision_trees(train_images_gray, train_labels, test_images_gray, test_labels [2, 4, 8, 16, 32, 64, 128])


