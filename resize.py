import os
# import constants
import numpy as np
from scipy import misc, ndimage
import imageio

d1 = 80
d2 = 80


def resize(image, dim1, dim2):
    return misc.imresize(image, (dim1, dim2))

def fileWalk(directory, destPath):
	print("starting")
	for subdir, dirs, files in os.walk(directory):
		print("in first loop")
		for file in files:
			print(file)
			# if len(file) <= 4 or file[-4:] != '.jpg':
			# 	continue
			pic = imageio.imread(os.path.join(subdir, file))
			# imshow(pic)
			dim1 = len(pic)
			dim2 = len(pic[0])
			if dim1 > dim2:
				pic = np.rot90(pic)
			picResized = resize(pic, d1, d2)
			misc.imsave(os.path.join(destPath, file), picResized)
			print('saving')



def main():
    # prepath = os.path.join(os.getcwd(), 'alldata')
    #
    # destPath = os.path.join(os.getcwd(), 'positive_images')

    prepath = os.path.join(os.getcwd(), 'negatives')

    destPath = os.path.join(os.getcwd(), 'opencv-haar-classifier-training/negative_images')


	#Do the resizing
    fileWalk(os.path.join(prepath), os.path.join(destPath))




if __name__ == '__main__':
    main()
