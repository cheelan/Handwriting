from scipy import sparse
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
import read_data
'''
Things to experiment with 
-Use pure black/white based on thresholding http://en.wikipedia.org/wiki/Otsu%27s_Method

for SVMs:
-Kernels
-C

KNN:
-k

Decision Trees:

'''
images, labels = read_data.read(range(10))
print "Read Raw Data"
sparse_images = preprocessing.scale(images)
#sparse_images = sparse.csr_matrix(sparse_images)

print "Made images sparse"

clf = svm.SVC(kernel='linear', cache_size=1000.)
print "initialized SVC"
clf.fit(sparse_images[:1000], labels[:1000])
print "Fit SVM"
guesses = clf.predict(sparse_images[10001:11001])
print metrics.classification_report(labels[10001:11001], guesses)
print guesses[:10]
print labels[10001:10011]

#cProfile.run('go()')
