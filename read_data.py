#From http://g.sweyla.com/blog/2012/mnist-numpy/
import os, struct
from array import array as pyarray
from pylab import *
from numpy import *
print "FAILURE"
def read(digits, dataset = "training", path = "."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset is "training":
        fname_img = 'data/train-images.txt'
        fname_lbl = 'data/train-labels.txt'
    elif dataset is "testing":
        fname_img = os.path.join(path, 'data/t10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'data/t10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in xrange(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

#print images[0]

#Draw "average" 2
#imshow(images.mean(axis=0), cmap=cm.gray)
#Draw a 2
#imshow(images[0], cmap=cm.gray)
#show()
#imshow(images[1], cmap=cm.gray)
#show()
#imshow(images[2], cmap=cm.gray)
#show()




