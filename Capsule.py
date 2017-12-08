import tensorflow as tf
import numpy as np
import math

POSE_MATRIX_X = 4
POSE_MATRIX_Y = 4
N_ROUTING_ITERS = 3
#My interpretation of https://openreview.net/pdf?id=HJWLfGWRb
LAMBDA = tf.placeholder(tf.float32)

class Capsule():
	def __init__(self, n_output_capsules, name, stride = 1, kernel_size = 1):
		self.stride = stride
		self.kernel_size = kernel_size
		self.n_output_capsules = n_output_capsules
		self.matrix_dim_size = POSE_MATRIX_X * POSE_MATRIX_Y
        self.layer_name = name

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
        	#FIXED FOR ALL PATCHES:
		n_i         = previous_layer.kernel_size * previous_layer.kernel_size * previous_layer.n_output_capsules
		n_c         = self.n_output_capsules
		R = tf.Variable(np.ones( (batch_size,n_c,n_i) ) / n_c )  #Dimensions opposite of paper (to simplify indexing), normally n_i,n_c matrix
        	beta_v, beta_a = tf.Variable(np.ones((1,n_c,1)), tf.Variable(np.ones((1,n_c)) #<= Nothing in the paper about how to init these!
		M = tf.Variable(np.ones( (batch_Size,n_c,n_i) ))
        	#####################
        	#PATCH SPECIFIC CODE:
        	#prev_capsules = previous_layer.capsules #Should have shape = [batch(b), width(x), height(h), channels(c)*(pose(h) + activations(a)) ]
             

        	#####TODO: Restore spatial information:
        	result = tf.extract_image_patches( prev_capsules
                                , [1,previous_layer.kernel_size,previous_layer.kernel_size,1]
                                , [1,previous_layer.stride,previous_layer.stride,1]
                                , [1,1,1,1]
                                , padding = 'VALID'
                                , name= 'extract_patches_{}'.format(self.layer_name)
                                )
                                    #[batch_size, x,y, parts, capsule, pose+activation]
        	patches = tf.reshape(result, [batch_size, patch_size, patch_size, -1,2,17])
        	patches = tf.reshape(resut,  [batch_size, patch_size, patch_size, -1,2,17])
	                           #patch_num, batch_size, x,y,capsule, pose+activation
        	patches = tf.transpose(result, [3,0, 1,2,4,5])
        	#There is a depth to space function that might be helpful
                                                            
            """
            #for patch in patches: use map_fn instead of for loop
            activations = previous_layer.activations #TODO: Just the activations for the active patch
            #Shape = batch_size, self.n_output_capsules
            Votes       = previous_layer.votes  #TODO: Must be just the votes for this patch
            #Assume shape of votes = (n_i, n_c, self.matrix_dim_size) until this module is complete
            """
    
	def route_patch(patch, M):
            activations = tf.expand_dims(a[:,:,-1],-1)   
            pose = tf.expand_dims(a[:,:,:-1])   
            for iteration in N_ROUTING_ITERS:
                M, S, tmp_activations = self.m_step(R, activations, Votes, beta_v, beta_a)
                R = self.e_step(M,S, tmp_activations, Votes, R)
	    return M, R

	new_tf.map_fn(route_patch, M 
                                                                     
######EVERYTHING BELOW HERE SHOULD BE CORRECT###################
	def m_step(self, r, a , V_prime, beta_v, beta_a): #FOR c (current layer) :
        #This should broadcast. R maintains its original shape
        #r shape = [batch(b),prev_layer(i),next_layer(c)]
        #a shape = [batch(a),prev_layer(i)] ===> expand dims makes this [batch(a), prev_layer(i), 1]
        r = r * tf.expand_dims(a, -1)
        #denominator shape should be [batch(b), next_layer(c)]
		denominator = tf.expand_dims(tf.squeeze(tf.reduce_sum(r,axis = 1)), -1) #<= so that it broadcasts for division
        #mu should have shape [batch(b), next_layer(c), pose_matrix(h)]
        #Numerator is sum of product between r(i) and V(i)(h)
        #r shape = [batch(b), prev_layer(i),next_layer(c)]
        #v shape = [batch(b), prev_layer(i),next_layer(c), pose_matrix(h)]
        numerator_mu = tf.squeeze(tf.reduce_sum(tf.expand_dims(r, -1) * V_prime, axis = 1))
        mu = numerator_mu/denominator
        #variance should have shape [batch(b), next_layer(c), pose_matrix(h)]
        # To calculate (V(i,h) - mu(h)**2, we need them to have the same number of dimensions:
        mu_num = tf.expand_dims(mu,1) #Add in 1 for the i dimension
        diff = tf.pow(V_prime - mu_num, 2)
        #diff has shape = [batch(b), prev_layer(i),next_layer(c), pose_matrix(h)]. Now we need to multiply it by r.
        #r has shape = [batch(b), prev_layer(i),next_layer(c)]. We need to expand this to multiply by diff and then reduce in the (i) dimension
        numerator_variance = tf.reduce_sum(tf.expand_dims(r, -1) * diff, 1)
        #numerator_variance should have shape = [batch(b), next_layer(c), pose_matrix(h)]
        variance = numerator_variance/denominator
		std = tf.sqrt(variance)
        #cost_h will have shape = [batch(b), next_layer(c), pose_matrix(h)]
        #std has the right shape, denominator shape = [batch(b), next_layer(c), 1]. Beta_v is [1,n_c,1]
		cost_h = (beta_v + tf.log(std)) * denominator
        #a_prime shape = [batch(b), next_layer(c)]
		a_prime = tf.nn.sigmoid(LAMBDA * (beta_a - tf.reduce_sum(cost_h, axis = 2)))
		return mu, std, a_prime

	def e_step(self, M, S, tmp_activations, votes,R):  #FOR i (prev layer) :
        #M     shape = [batch(b), next_layer(c), pose_matrix(h)]
        #S     shape = [batch(b), next_layer(c), pose_matrix(h)]
        #votes shape = [batch(b), prev_layer(i),next_layer(c), pose_matrix(h)]
        #P     shape = [batch(b), next_layer(c), prev_layer(i)] <= Unknown!
        part_1 = 1./tf.expand_dims(tf.reduce_prod(2*math.pi*tf.pow(S,2), axis = 2),1)
        # part_1 shape = [batch(b), 1, next_layer(c)]
        part_2_num = tf.pow(votes-tf.expand_dims(M,1),2)
        # part_2_num shape = [batch(b), prev_layer(i),next_layer(c), pose_matrix(h)]
        part_2_den = 2*tf.expand_dims(tf.pow(S,2),1)
        # part_2_den shape = [batch(b), 1, next_layer(c) , pose_matrix(h)]
        part_2 = tf.exp(-tf.reduce_prod(part_2_num/part_2_den, axis = 3)
        #part_2 shape = [batch(b),prev_layer(i),next_layer(c)]
        P = part_1 * part_2
        #p shape = [batch(b),prev_layer(i),next_layer(c)]
        #tmp_activations shape = [batch(b), next_layer(c)]
        #R shape = [batch(b), next_layer(c), prev_layer(i)]
        #Note the paper indexes the numerator with (j). But (j) randomly appeared from nowhere. It only makes sense if its (i), so this code is assuming (i).
        R_num = tf.expand_dims(tmp_activations,1)*P
        # R_num shape = [batch(b), prev_layer(i),next_layer(c)]
        R_den = tf.expand_dims(tf.reduce_sum( tf.expand_dims(tmp_activations,1)*P, axis = 2),-1) #Not sure if I am reducing by the right
        # R_den shape = [batch(b), prev_layer(i), 1 ]
        R = R_num/R_den
        #R shape = [batch(b), next_layer(c), prev_layer(i)]
        return R





