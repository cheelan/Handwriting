def otsu(image):
	total = len(image)
	histogram = [0] * 256
	for i in image:
		histogram[int(i)] += 1
	sumA = 0
	for i in range(len(histogram)):
		sumA += i * histogram[i]
	sumB = 0
	wB = 0
	wF = 0
	maximum = 0
	threshold = 0
	for i in range(len(histogram)):
		wB += histogram[i]
		if wB == 0:
			continue
		wF = total - wB
		if wF == 0:
			break
		sumB += i * histogram[i]
		mB = sumB / wB
		mF = (sumA - sumB) / wF
		between = wB * wF * (mB - mF) * (mB - mF)
		if between > maximum:
			maximum = between
			threshold = i

	newImage = [0.] * total

	for i in range(total):
		if image[i] > threshold:
			newImage[i] = 1.

	return newImage
