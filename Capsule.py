import tensorflow as tf
import numpy as np
import math

POSE_MATRIX_X = 4
POSE_MATRIX_Y = 4
N_ROUTING_ITERS = 3
#My interpretation of https://openreview.net/pdf?id=HJWLfGWRb
LAMBDA = 1#tf.placeholder(tf.float32)



class Conv2D():
    def __init__(self, n_output_channels, name , stride = 1, kernel_size = 1):
        self.kernel_size = kernel_size
        self.n_output_capsules = n_output_channels
        self.name = name
        self.stride = stride
        self.type = 'conv'
        self.call = lambda x: tf.layers.conv2d( x
                                , n_output_channels
                                , [kernel_size,kernel_size]
                                , [stride,stride]
                                , name = name)

    def __call__(self, prev_layer):
        self.activations =  self.call(prev_layer)
        return self


class Capsule():
    def __init__(self, n_output_capsules, name, stride = 1, kernel_size = 1):
        self.type = 'capsule'
        self.stride = stride
        self.kernel_size = kernel_size
        self.n_output_capsules = n_output_capsules
        self.matrix_dim_size = POSE_MATRIX_X * POSE_MATRIX_Y
        self.layer_name = name

    def _init_em(self):
        self.count = 0
        self.n_i = self.kernel_size * self.kernel_size * self.n_output_capsules
        self.n_c = self.n_output_capsules
        self.R = tf.Variable(np.ones( (self.batch_size,self.n_i,self.n_c) ) / self.n_c )
        self.R = tf.to_float(self.R)
        self.beta_v, self.beta_a = tf.Variable(np.ones((1,self.n_c,1))), tf.Variable(np.ones((1,self.n_c))) #<= Nothing in the paper about how to init these!
        self.beta_v = tf.to_float(self.beta_v)
        self.beta_a = tf.to_float(self.beta_a)
        self.M = []

    def __call__(self, previous_layer):
        if(previous_layer.type == 'capsule'):
            self.batch_size = int(previous_layer.activations.get_shape()[0])
            self._init_em()
            prev_shape = previous_layer.activations.get_shape()
            self.activations = self.route_em(previous_layer)
            new_shape = (self.batch_size,prev_shape[1]//self.stride, prev_shape[2]//self.stride, self.n_output_capsules)
            self.activations.set_shape( (new_shape) )
        elif(previous_layer.type == 'conv'):
            self.batch_size = int(previous_layer.activations.get_shape()[0])
            self._init_em()
            #No routing necessary - regular convolution with 17 output channels time the number of output channels
            n_output_channels = self.n_output_capsules * self.matrix_dim_size
            self.output = tf.layers.conv2d(previous_layer.activations
                                        , n_output_channels
                                        , [self.kernel_size,self.kernel_size]
                                        ,[self.stride,self.stride]
                                        , activation = None)

            tmp_pose = self.output[:,:,:,previous_layer.n_output_capsules:]
            tmp_activations = tf.nn.sigmoid(self.output[:,:,:,-previous_layer.n_output_capsules:])
            self.activations = self.primary_conv(tmp_activations,tmp_pose,self.route_patch, int(previous_layer.activations.get_shape()[-1]))
            prev_shape = previous_layer.activations.get_shape()
            new_shape = (self.batch_size,prev_shape[1]//self.stride, prev_shape[2]//self.stride, self.n_output_capsules)
            self.activations.set_shape( new_shape )
        else:
            raise TypeError("Input must be either a tensor or capsule. Input type given : {}".format(type(previous_layer)))
        return self
			        

    def route_patch(self,activations, votes, m, r):
            n_dims = len(activations.get_shape())
            activations = tf.reshape(activations,[self.batch_size ,-1])
            """
            print("Activations are {}".format(activations))
            print("votes are {}".format(votes))
            print("m are {}".format(m))
            print("r are {}".format(r))
            """
            assert n_dims == 4, "Patch must be 4 dimenional but is {}".format(n_dims)
            for iteration in range(N_ROUTING_ITERS):
                
                m, s, activations = self.m_step(r, activations, votes, self.beta_v, self.beta_a)
                
                    
                    

                r = self.e_step(m,s, activations, votes, r)
            return activations, m #M = pose

    def route_em(self, previous_layer):
        self.activations = self.custom_conv(previous_layer, self.kernel_size, self.kernel_size, self.stride, self.stride, self.route_patch)
        return self.activations


    def custom_conv(self,input_layer, k_r, k_c, s_r, s_c, op):
        kernel_size_rows = k_r #3
        kernel_size_cols = k_c #3
        strides_rows = s_r #2
        strides_cols = s_c #2
        ksizes = [1, kernel_size_rows, kernel_size_cols, 1]
        strides = [1, strides_rows, strides_cols, 1]
        rates = [1, 1, 1, 1]
        padding = 'VALID'
        image_patches = tf.extract_image_patches(input_layer.activations, ksizes, strides, rates, padding)
        shape = [int(x) for x in image_patches.get_shape()]

        if(len(shape) == 4):
            prev_shape = input_layer.activations.get_shape()
            batch_size = int(prev_shape[0])
            nr = shape[1]
            nc = shape[2]
            n_channels = int(prev_shape[3]) #Make sure this represents the number of incoming capsules. Not 17* n-capsules
        else:
            raise NotImplementedError("No support for >4d tensor")

        feature_map_patches = []
        index = 0
        votes = tf.random_normal([batch_size, k_r*k_c*n_channels,self.n_output_capsules,16])
        for i in range(nr):
            print("row {}/{}".format(i,nr))
            feature_map_patches_cols = []
            for j in range(nc):
                print("col {}/{}".format(j,nc))
                batch_patch = []
                for b in range(batch_size):
                    patch = tf.expand_dims(tf.reshape(image_patches[b,i,j,], [kernel_size_rows, kernel_size_cols, n_channels]), 0)
                    batch_patch.append(patch)
                self.M.append(tf.Variable(np.ones( (self.batch_size,self.n_c,16) )))
                patch = tf.concat(batch_patch, axis = 0)
                print("THE PATCH {}".format(patch))
                
                activations, self.M[index] = op(patch, votes, self.M[index], self.R)
                
                feature_map_patches_cols.append(activations)
                index+=1
            col = tf.concat(feature_map_patches_cols, axis = 2)
            feature_map_patches.append(col)
        return tf.concat(feature_map_patches, axis = 1)


    def primary_conv(self, activations,m,op, input_channels):
        #todo: Consolidate this and custom_conv into one function
        kernel_size_rows = self.kernel_size
        kernel_size_cols = self.kernel_size
        strides_rows = self.stride
        strides_cols = self.stride
        ksizes = [1, kernel_size_rows, kernel_size_cols, 1]
        strides = [1, strides_rows, strides_cols, 1]
        rates = [1, 1, 1, 1]
        padding = 'VALID'
        activation_patches = tf.extract_image_patches(activations, ksizes, strides, rates, padding)
        pose_patches = tf.extract_image_patches(m, ksizes, strides, rates, padding)
        shape = [int(x) for x in activation_patches.get_shape()]
        if(len(shape) == 4):
            batch_size = int(activations.get_shape()[0])
            nr = shape[1]
            nc = shape[2]
            n_a_channels = int(activations.get_shape()[3]) #Make sure this represents the number of incoming capsules. Not 17* n-capsules
            n_p_channels = int(m.get_shape()[3])
        else:
            raise NotImplementedError("No support for >4d tensor")


        feature_map_patches = []
        index = 0
        votes = tf.random_normal([batch_size, self.kernel_size*self.kernel_size*input_channels,self.n_output_capsules*16])
        for i in range(nr):
            print("row {}/{}".format(i,nr))
            feature_map_patches_cols = []
            for j in range(nc):
                print("col {}/{}".format(j,nc))
                batch_patch_a, batch_patch_m = [], []
                for b in range(batch_size):
                    patch_a = tf.expand_dims(tf.reshape(activation_patches[b,i,j,], [kernel_size_rows, kernel_size_cols, n_a_channels]), 0)
                    patch_m = tf.expand_dims(tf.reshape(pose_patches[b,i,j,], [kernel_size_rows, kernel_size_cols, n_p_channels]), 0)
                    batch_patch_a.append(patch_a)
                    batch_patch_m.append(patch_m)
                patch_a = tf.concat(batch_patch_a, axis = 0)
                patch_m = tf.concat(batch_patch_m, axis = 0)
                activations, _ = op(patch_a, votes, patch_m, self.R)
                feature_map_patches_cols.append(tf.reshape(activations,[self.batch_size, self.kernel_size, self.kernel_size,self.n_output_capsules]))
                index+=1
            col = tf.concat(feature_map_patches_cols, axis = 2)
            feature_map_patches.append(col)     
        return tf.concat(feature_map_patches, axis = 1)


                                                                                 
    def m_step(self, r, a , V_prime, beta_v, beta_a): #FOR c (current layer) :
        #This should broadcast. R maintains its original shape
        #r shape = [batch(b),prev_layer(i),next_layer(c)]
        #a shape = [batch(a),prev_layer(i)] ===> expand dims makes this [batch(a), prev_layer(i), 1]
        r = r * tf.expand_dims(a, -1)
        #denominator shape should be [batch(b), next_layer(c)]
        denominator = tf.expand_dims(tf.reduce_sum(r,axis = 1), -1) #<= so that it broadcasts for division
        
        #mu should have shape [batch(b), next_layer(c), pose_matrix(h)]
        #Numerator is sum of product between r(i) and V(i)(h)
        #r shape = [batch(b), prev_layer(i),next_layer(c)]
        #v shape = [batch(b), prev_layer(i),next_layer(c), pose_matrix(h)]

        numerator_mu = tf.reduce_sum(tf.expand_dims(r, -1) * V_prime, axis = 1)
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
        print(a_prime.get_shape())
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
        part_2 = tf.exp(-tf.reduce_prod(part_2_num/part_2_den, axis = 3))
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





