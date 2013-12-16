import read_data
import math
import cProfile
import pylab as pl
import otsu
import numpy
import Image
#See http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html
from sklearn import cross_validation
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

#Read dataset
train_images_gray, train_labels = read_data.read(range(10), 'training')
test_images_gray, test_labels = read_data.read(range(10), 'testing')
print "Done reading data"

def generate_pngs():
  for x in xrange(10):
    test = train_images_gray[x]
    test2 = []
    for i in xrange(len(test)):
      test2.append(numpy.uint8(255 - test[i]))
    test2 = numpy.array(test2)
    test2.shape = (28,28)
    print test2
    img = Image.fromarray(test2, 'L')
    img.save('report/' +str(x) + '.png')

def convert_bw(images):
  bw = []
  for i in images:
    bw.append(numpy.array(otsu.otsu(i)))
  return numpy.array(bw)

train_images_bw = convert_bw(train_images_gray)
test_images_bw = convert_bw(test_images_gray)



def svms(images, labels, test_images, test_labels, degrees):
  for d in degrees:
    clf = svm.SVC(kernel="poly", degree=d)
    #scores = cross_validation.cross_val_score(clf, images, labels, cv=5)
    scores = clf.fit(images, labels).score(test_images, test_labels)
    print "Testing SVM d = " + str(d) + " " + str(scores)
    #print "Testing SVM kernel = " + k + " c = " + str(training_penalty) + " " + str(scores)

#Doesn't necessarily use all of max_depth. TDIDT stops once all training examples are classified correctly
def decision_trees(images, labels, test_images, test_labels, depths, showimage=False):
    for d in depths:
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
# svms(train_images_bw[:10000], train_labels[:10000], test_images_bw, test_labels, [1,2,3,4,5])
# random_forests(train_images_bw[:10000], train_labels[:10000], test_images_bw, test_labels, [1, 10, 50, 100], [2, 4, 8, 16, 32], False)
# decision_trees(train_images_bw[:10000], train_labels[:10000], test_images_bw, test_labels, [2, 4, 8, 16, 32, 64, 128])

# print "GRAYSCALE"
# svms(train_images_gray[:10000], train_labels[:10000], test_images_gray, test_labels, [1,2,3,4,5])
# random_forests(train_images_gray[:10000], train_labels[:10000], test_images_gray, test_labels, [1, 10, 50, 100], [2, 4, 8, 16, 32], False)
# decision_trees(train_images_gray[:10000], train_labels[:10000], test_images_gray, test_labels, [2, 4, 8, 16, 32, 64, 128])
random_forests(train_images_bw[:10000], train_labels[:10000], test_images_bw, test_labels, [100], [32], True)
random_forests(train_images_gray[:10000], train_labels[:10000], test_images_gray, test_labels, [100], [32], True)

