import jtov
import numpy
import read_data
import otsu
from sklearn import cross_validation
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

USEBW = True

def convert_bw(images):
  bw = []
  for i in images:
    bw.append(numpy.array(otsu.otsu(i)))
  return numpy.array(bw)

def classify(image):
    train_images_gray, train_labels = read_data.read(range(10), 'training')
    if USEBW:
        train_images_bw = convert_bw(train_images_gray)
        test = otsu.otsu(numpy.array(jtov.jtov(image)))
        clf = svm.SVC(kernel="poly", degree=1)
        clf.fit(train_images_bw, train_labels)
    else:
        test = numpy.array(jtov.jtov(image))
        clf = svm.SVC(kernel="poly", degree=2)
        clf.fit(train_images_gray, train_labels)
	print test
	print clf.predict(test)

classify("photo.JPG")