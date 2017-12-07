#import tensorflow as tf
import numpy as np

POSE_MATRIX_X = 4
POSE_MATRIX_Y = 4
N_ROUTING_ITERS = 3
#My interpretation of https://openreview.net/pdf?id=HJWLfGWRb
LAMBDA = tf.placeholder(tf.float32)
class Capsule():
	def __init__(self, n_output_capsules, stride = 1, kernel_size = 1):
		self.stride = stride
		self.kernel_size = kernel_size
		self.n_output_capsules = n_output_capsules
		self.matrix_dim_size = POSE_MATRIX_X * POSE_MATRIX_Y

	def __call__(self, previous_layer):
		if(type(previous_layer) == Capsule):
			#Perform routing
			prev_kernel_size = previous_layer.kernel_size

		elif(type(previous_layer)  == tf.Tensor):
			#No routing necessary - regular convolution with 17 output channels time the number of output channels
			n_output_channels = self.n_output_capsules * self.matrix_dim_size
			output = tf.layers.conv2d(previous_layer, n_output_channels):	
			#Need to slice outputs into activations and votes  
		else:
			raise TypeError("Input must be either a tensor or capsule. Input type given : {}".format(type(previous_layer)))

		return self
			
	def route_em(self, previous_layer):		
		#This has to be applied to all conv patches
		
		activations = previous_layer.activations #TODO: Just the activations for the active patch
		#Shape = self.n_output_capsules
		Votes       = previous_layer.votes  #TODO: Must be just the votes for this patch
		#Assume shape of votes = (n_c, self.matrix_dim_size, n_i) until this module is complete
		n_i         = previous_layer.kernel_size * previous_layer.kernel_size * previous_layer.n_output_capsules
		n_c         = self.n_output_capsules
		R = tf.Variable(np.ones( (n_c,n_i) ) / n_c )  #Dimensions opposite of paper (to simplify indexing), normally n_i,n_c matrix
		M = tf.Variable(np.ones( (n_c,n_r,self.matrix_dim_size) ) 
		S = tf.Variable(np.ones( (n_c,n_r,self.matrix_dim_size) )
		for iteration in N_ROUTING_ITERS:
			for c in range(n_c):
				#For all c (output nodes)
				M, S, tmp_activations = tf.map_fn(lambda r,a,v: self.m_step(r,a,v) ,(R activations, Votes), (tf.float32,tf.float32,tf.float32))	#<=Same as for all c
				#For all i (input nodes)
				R = tf.transpose(R, [1,0])  #make the ith dimension the first so we iterate over it
				R = tf.map_fn(lambda x: self.e_step(M, S, tmp_activations, votes,R), R, tf.float32) 
				R = tf.transpose(R, [0,1]) 
				
	def m_step(self, r, a , V_prime): #FOR c (current layer) :
		r = r * a  #<= a and r should e the same size (Number of input weights (subscripted by i))
		denominator = tf.reduce_sum(r,axis = 0)
		mu = tf.map_fn(lambda x: tf.div(tf.reduce_sum(r*x[0],axis = 0), denominator), V_prime, tf.float32) #Will have a shape of h (16)
		variance = tf.map_fn(lambda x: tf.div(tf.reduce_sum(r*tf.pow((x[0]-x[1]),2),axis = 0), denominator), V_prime, tf.float32) #Will have a shape of h (16) 
		std = tf.sqrt(variance)
		cost_h = (Beta_v + tf.log(std)) * denominator
		a_prime = tf.nn.sigmoid(LAMBDA * (Beta_a - tf.reduce_sum(cost_h, axis = 0)))
		return mu, std, a_prime	

	def e_step(self, M, S, tmp_activations, votes,R):  #FOR i (prev layer) :
			





