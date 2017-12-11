import tensorflow as tf
import numpy as np
from Capsule import Capsule, Conv2D



class Model():
	def __init__(self,batch_size, img_dims):
		self.input = tf.placeholder(tf.float32, shape = [batch_size, img_dims, img_dims, 3])	
		conv = Conv2D(32, kernel_size = 5, stride = 2, name = "conv1")(self.input)
		print("Starting primary caps")
		primary_caps = Capsule(n_output_capsules = 32
							 , stride = 1
							 , kernel_size = 2
							 , name = "primary_caps")(conv)


		print("Starting conv caps 1")
		conv_caps_1 = Capsule( n_output_capsules = 32 
							 ,  stride = 2
							 , kernel_size = 3
							 , name = "conv_caps_1"
							)(primary_caps)

		print(conv_caps_1)

		self.conv_caps_2 = Capsule(n_output_capsules = 32 
							 ,  stride = 1
							 , kernel_size = 3
							 , name = "conv_caps_2"
							)(conv_caps_1)







if __name__ == '__main__':
	model = Model(1, 16)
	with tf.Session() as sess:
		input_ar = np.random.random( (1, 32, 32, 3) )
		sess.run(tf.global_variables_initializer())
		print(sess.run(model.conv_caps_2.activations, feed_dict = {model.input: input_ar}).shape)
