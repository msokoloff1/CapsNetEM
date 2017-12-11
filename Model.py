import tensorflow as tf
import numpy as np
from Capsule import Capsule, Conv2D



class Model():
	def __init__(self,batch_size, img_dims, n_classes):
		self.input = tf.placeholder(tf.float32, shape = [batch_size, img_dims, img_dims, 3])	
		self.label_ph = tf.placeholder(tf.float32, shape = [batch_size, n_classes])
		self.margin = tf.placeholder(tf.float32)
		conv = Conv2D(32, kernel_size = 5, stride = 2, name = "conv1")(self.input)
		
		primary_caps = Capsule(n_output_capsules = 32
							 , stride = 1
							 , kernel_size = 1
							 , name = "primary_caps")(conv)

		
		conv_caps_1 = Capsule( n_output_capsules = 32
							 ,  stride = 2
							 , kernel_size = 3
							 , name = "conv_caps_1"
							)(primary_caps)

		

		conv_caps_2 = Capsule(n_output_capsules = 32 
							 ,  stride = 1
							 , kernel_size = 3
							 , name = "conv_caps_2"
							)(conv_caps_1)

		self.class_capsules = Capsule(n_output_capsules = n_classes
							  , stride = 1
							  , kernel_size = 1
							  , name = 'output')(conv_caps_2)

		#TODO: Replace this with the Coordinate Addition
		output = tf.reshape(self.class_capsules,[batch_size, 1])
		outout = tf.layers.dense(output, 5, name = 'output', activation = tf.nn.sigmoid)
		self.prediction = out
		self.build_loss(n_classes)

	def build_loss(self, n_classes):
		mask = tf.one_hot(tf.argmax(self.label_ph, axis = 1), n_classes)
		mask = mask < 1
		mask = tf.to_float(mask)
		L = mask*tf.reduce_max(0,self.margin - (self.label_ph - self.prediction), axis = 1)
		Loss = tf.reduce_sum(L)
		return loss








if __name__ == '__main__':
	model = Model(1, 32, 5)
	with tf.Session() as sess:
		input_ar = np.random.random( (1, 32, 32, 3) )
		sess.run(tf.global_variables_initializer())
		print(sess.run(model.class_capsules.activations, feed_dict = {model.input: input_ar}).shape)
