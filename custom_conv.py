import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize, imsave
import math


#Need to be sure that when we squash 5d tensor to 4d we can recover the 5th dim structure

def custom_conv(input_layer, k_r, k_c, s_r, s_c, op):
	kernel_size_rows = k_r #3
	kernel_size_cols = k_c #3
	strides_rows = s_r #2
	strides_cols = s_c #2
	ksizes = [1, kernel_size_rows, kernel_size_cols, 1]
	strides = [1, strides_rows, strides_cols, 1]
	rates = [1, 1, 1, 1]
	padding = 'VALID'
	print("Input image {}".format(input_layer.get_shape()))
	image_patches = tf.extract_image_patches(input_layer, ksizes, strides, rates, padding)
	print("Image patches {}".format(image_patches.get_shape()))
	shape = [int(x) for x in image_patches.get_shape()]

	if(len(shape) == 4):
		batch_size = int(input_layer.get_shape()[0])
		nr = shape[1]
		nc = shape[2]
		n_channels = int(input_layer.get_shape()[3])
	else:
		raise NotImplementedError("No support for >4d tensor. Need to add this for capsules")

	feature_map_patches = []
	for i in range(nr):
		feature_map_patches_cols = []
		for j in range(nc):
			batch = []
			for b in range(batch_size):
				patch = tf.expand_dims(tf.reshape(image_patches[b,i,j,], [kernel_size_rows, kernel_size_cols, n_channels]), 0)
				batch.append(patch)
			patch = tf.concat(batch, axis = 0)
			#TODO: Apply op once batch has been concatenated
			#Need to restore 5th dim here and then run op to get resulting capsule
			feature_map_patches_cols.append(patch)
		col = tf.concat(feature_map_patches_cols, axis = 2)
		feature_map_patches.append(col)		
	return tf.concat(feature_map_patches, axis = 1)
		




if __name__ == '__main__':
	#Test to see if it works on a batch
	imgs = np.concatenate([np.expand_dims(imresize(imread(p), (400,400,3)),0) for p in ['girl.jpg','car.jpg']], -1)
	img_tensor = tf.stack(imgs)
	sess = tf.Session()
	result = np.squeeze(sess.run(custom_conv(img_tensor, 50,50,50,50,None)))
	print(result.shape)
	imsave('first.jpg', result[:,:,:3])
	imsave('second.jpg', result[:,:,3:])
	sess.close()

