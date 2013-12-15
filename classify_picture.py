import jtov
import numpy
import read_data
from sklearn import cross_validation
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

def classify(image):
	train_images_gray, train_labels = read_data.read(range(10), 'training')
	test = numpy.array(jtov.jtov(image))
	print test
	clf = svm.SVC(kernel="poly", degree=2)
	clf.fit(train_images_gray, train_labels)
	print clf.predict(test)

classify("photo.JPG")