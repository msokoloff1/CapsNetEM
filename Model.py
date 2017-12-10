import tensorflow as tf
import numpy as np
from Capsule import Capsule



class Model():
	def __init__(self,batch_size, img_dims):

		self.input = tf.placeholder(tf.float32, shape = [batch_size, img_dims, img_dims, 3])	
		conv = tf.layers.conv2d(self.input, 32, [5,5],[2,2], activation = tf.nn.relu)
		print(type(conv))
		primary_caps = Capsule(n_output_capsules = 32
							 , stride = 1
							 , kernel_size = 1
							 , name = "primary_caps")(conv)
		print(type(primary_caps))
		exit(0)

		conv_caps_1 = Capsule( n_output_capsules = 32 
							 ,  stride = 2
							 , kernel_size = 3
							 , name = "conv_caps_1"
							)(primary_caps)





if __name__ == '__main__':
	model = Model(1, 128)