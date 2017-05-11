import copy, numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def tanh(x):
    output = (1-np.exp(-2*x)/(1+np.exp(-2x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

def tanh_output_to_derivative(output):
    return 1-output*output


# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]


# input variables
alpha = 0.1
input_dim = 1
hidden_dim = 3
output_dim = 2


# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(10):
    
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

	z_t1 = sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
	r_t1 = sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
	c_t1 = tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
	s_t1 = (np.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

        # output layer (new binary representation)
#        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
#
#	z_t2 = sigmoid(U[3].dot(x_e) + W[3].dot(s_t1_prev) + b[3])
#	r_t2 = sigmoid(U[4].dot(x_e) + W[4].dot(s_t1_prev) + b[4])
#	c_t2 = tanh(U[5].dot(x_e) + W[5].dot(s_t1_prev * r_t1) + b[5])
#	s_t2 = (np.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

        layer_2 = sigmoid(np.dot(s_t1,synapse_1))

        # did we miss?... if so, by how much?

        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
	#output_error = y - s_t1
	#output_delta.append((output_error)*sigmoid_output_to_derivative(st1))
	#overallError += np.abs(s_t1[0])

        # decode estimate so we can print it out

        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        #d[binary_dim - position - 1] = np.round(s_t1[0][0])
        
        # store hidden layer so we can use it in the next timestep
        #layer_1_values.append(copy.deepcopy(layer_1))
	z_t1_values.append(copy.deepcopy(z_t1)
	r_t1_values.append(copy.deepcopy(r_t1)
	c_t1_values.append(copy.deepcopy(c_t1)
	s_t1_values.append(copy.deepcopy(s_t1)
    
    #future_layer_1_delta = np.zeros(hidden_dim)
    future_s_delta = np.zeros(hidden_dim)	
    
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        #layer_1 = layer_1_values[-position-1]
	s_t1 = s_t1_values[-position-1]
        #prev_layer_1 = layer_1_values[-position-2]
        prev_s_1 = s_t1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
	
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
	
	s_t2_delta = s_t1_delta
	s_t1_delta = s_t0_delta
	s_t0 = 
	delta_out = s_t1_delta + s_t2_delta
	b0_delta = delta_out*(c_t-z_t)*sigmoid_output_to_derivative(z_t) 
	b2_delta = delta_out*c_t*tanh_output_to_derivative(c_t)
	b1_delta = b2_delta.dot(U.T) * s_t0 * sigmoid_output_to_derivative(r_t)

	W_delta[2] = np.dot(X.T,b2_delta)
	U_delta[2] = np.dot((s_t0*r_t).T,b2_delta)
	W_delta[1] = np.dot(X.T,b1_delta)
	U_delta[1] = np.dot(s_t0.T,b1_delta)
	W_delta[0] = np.dot(X.T,b0_delta)
	U_delta[0] = np.dot(z_t.T,b0_delta)
	s_t0_delta = (1-z_t)*delta_out + b0_delta.dot(U0.T) + r_t*(b2_delta.dot(U2.T) + b1_delta.dot(U1.T) 
	

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
	U_update
        future_layer_1_delta = layer_1_delta
    

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(a_int) + " + " + str(b_int) + " = " + str(out)
        print "------------"

        
