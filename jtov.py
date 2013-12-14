import os, sys
import Image
import otsu

def jtov(filename):
	f = Image.open(filename)
	#f.show()
	newF = f.crop((0, 0, 240, 240))
	finalF = newF.convert('L')
	finalF = finalF.resize((28, 28))
	finalF.save("gray" + filename)
	l = list(finalF.getdata())
	print len(l)
	finalL = [0] * len(l)
	for i in range(len(l)):
		finalL[i] = 255 - l[i]
	print otsu.otsu(finalL)


jtov("photo.JPG")