import os, sys
import Image

def jtov(filename):
	f = Image.open(filename)
	f.show()
	newF = f.crop((0, 0, 240, 240))
	finalF = newF.convert('L')
	finalF = finalF.resize((28, 28))
	finalF.save("gray" + filename)
	print list(finalF.getdata())


jtov("photo.JPG")