import read_data
import subprocess

def learn(train_file, model_file="svm_model.txt"):
  args = ['./svm_learn', train_file, model_file]
  #args = ['svm_learn.exe', train_file, model_file]
  subprocess.call(args)
  print("Done SVM Learn")

def classify(test_file, output_file, model_file="svm_model.txt"):
  args = ['./svm_classify', test_file, model_file, output_file]
  subprocess.call(args)
  print("Done SVM Classify")
  ans = []
  with open(output_file) as f:
    for line in f.readlines():
      ans.append(float(line)) #The margin of the classification
  return ans

#Outputs a string in svmlight format: label id1:f1
def to_svm_format(matrix, label):
  vector = str(label) + " "
  id = 0
  for row in matrix:
    for i in xrange(len(row)):
      if row[i] != 0:
        vector += str(id) + ":" + str(row[i]) + " " #Note: no normalization
      id += 1
  return vector[:-1] #Skip the last whitespace

def load_data():
  images1, labels1 = read_data.read([1])
  images2, labels2 = read_data.read([2])
  ones = len(images1)
  with open('svm_train12.txt', 'w') as f:
    for i in images1:
      f.write(to_svm_format(i, -1) + "\n")
    print "Done 1s"
    for i in images2:
      f.write(to_svm_format(i, 1) + "\n")
    print "Done 2s"
  

  twos = len(images2)

  return ones, twos

def get_accuracy(real, guesses):
  correct = 0
  for i, ans in enumerate(real):
    label = 0
    if guesses[i] > 0:
      label = 1
    else:
      label = -1
    if ans == label:
      correct += 1
  return float(correct) / float(len(real))

#ones, twos = load_data()
ones = 6742
twos = 12700 - 6742
learn("svm_train12.txt")
print "Done learning"
guesses = classify("svm_train12.txt", "output.txt")
real = [-1] * ones + [1] * twos
print len(real)
print len(guesses)

print get_accuracy(real, guesses)
#images, labels = read_data.read([2])
#s = to_svm_format(images[0], 1)
#print(len(s))




