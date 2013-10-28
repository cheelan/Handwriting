import read_data


#Outputs a string in svmlight format: label id1:f1
def to_svm_format(matrix, label):
  vector = str(label) + " "
  for row in matrix:
    for i in xrange(len(row)):
      if row[i] != 0:
        vector += str(i) + ":" + str(row[i]) + " " #Note: no normalization
  return vector[:-1]

images, labels = read_data.read([2])
s = to_svm_format(images[0], 1)
print(len(s))




