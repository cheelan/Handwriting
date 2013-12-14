import os, sys
import Image

def jtov(filename):
	f = Image.open(filename)
	newF = f.crop((0, 0, 240, 240))
	finalF = newF.convert('L')
	finalF.save("gray" + filename)


jtov("photo.JPG")