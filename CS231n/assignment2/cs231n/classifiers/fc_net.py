from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.num_classes = num_classes

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1 = weight_scale*np.random.randn(input_dim, hidden_dim)
        W2 = weight_scale*np.random.randn(hidden_dim, num_classes)

        b1 = np.zeros(hidden_dim)
        b2 = np.zeros(num_classes)
        
        self.params = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # affine + RELU Layer 1
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        out_l1, cache_l1 = affine_relu_forward(X, W1, b1)   
        # affine + RELU Layer 2
        out_l2, cache_l2 = affine_relu_forward(out_l1, W2, b2)

        scores = out_l2


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # loss and dout by softmax
        loss, dout = softmax_loss(scores, y)

        # add loss with 1/2 * reg * (magnitude of W1 + magnitude of W2)
        loss += 0.5 * self.reg*((W1 * W1).sum() + (W2 * W2).sum())

        # gradients flow backwards from each affine relu layers
        dx_l2, dw2, db2 = affine_relu_backward(dout, cache_l2)
        dx_l1, dw1, db1 = affine_relu_backward(dx_l2, cache_l1)


        # Don't forget to add dLoss/dW2 and dLoss/dW1
        dw2 += self.reg*W2
        dw1 += self.reg*W1

        # set all the gradients to grads
        grads = {'b2': db2, 'b1': db1, 'W1': dw1, 'W2': dw2}
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.hidden_dims_size = len(hidden_dims)

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        # W1 = weight_scale*np.random.randn(input_dim, hidden_dim)
        # W2 = weight_scale*np.random.randn(hidden_dim, num_classes)

        # b1 = np.zeros(hidden_dim)
        # b2 = np.zeros(num_classes)


        # num_hid_layers = number of hidden layers
        num_hid_layers = len(hidden_dims)

        input_size = input_dim

        # For the hidden layers before final output
        for layer_idx in range(num_hid_layers):
            # get name = W + 
            name_w = 'W' + str(layer_idx+1)
            name_b = 'b' + str(layer_idx+1)

            self.params[name_w] = np.random.normal(loc = 0, scale = weight_scale, \
                                    size = (input_size, hidden_dims[layer_idx]))

            self.params[name_b] = np.zeros(shape = (hidden_dims[layer_idx], ))

            # if BN is used
            if (normalization == 'batchnorm'):
                name_gamma = 'gamma' + str(layer_idx+1)
                name_beta = 'beta' + str(layer_idx+1)

                self.params[name_gamma] = np.ones(hidden_dims[layer_idx])

                self.params[name_beta] = np.zeros(hidden_dims[layer_idx])


            input_size = hidden_dims[layer_idx]

        # for final layer before output
        self.params['W' + str(num_hid_layers+1)] = np.random.normal(0, weight_scale, (input_size, num_classes))

        self.params['b' + str(num_hid_layers+1)] = np.zeros(shape = (num_classes, ))


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)



    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnor maccparams and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode


        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        
        # First input into the network is X
        input_ = X
        # out_affine, cache_affine = [], []
        # out_bn, cache_bn = [], []
        # out_relu, cache_relu = [], []

        cache_affine, cache_bn, cache_relu = {}, {}, {}

        # for hidden layers before output
        for layer_idx in range(self.num_layers-1):
           #print (layer_idx)
            # get name of params W and b

            # get index for each layer
            index_str = str(layer_idx+1)

            name_w = 'W'+ index_str
            name_b = 'b'+ index_str

            W = self.params[name_w]
            b = self.params[name_b]
            # calculate the output of the present layer

            # First step: affine forward
            out_, cache_affine[index_str] = affine_forward(input_, W, b) 


            # Second: Batch Norm layer
            if (self.normalization=='batchnorm'):
                name_gamma = 'gamma'+ index_str
                name_beta = 'beta'+ index_str

                out_bn_, cache_bn[index_str] = batchnorm_forward(out_, self.params[name_gamma], self.params[name_beta], self.bn_params[layer_idx])
            
            # Third : ReLU    
                out_relu_, cache_relu[index_str] = relu_forward(out_bn_)



            else: # ifnot using BN then input of Relu is output of affine_forward
                out_relu_, cache_relu[index_str] = relu_forward(out_)
            
                 
            input_ = out_relu_.copy()
        

        # For layer right before the output
        name_w = 'W'+ str(self.hidden_dims_size+1)
        name_b = 'b'+ str(self.hidden_dims_size+1)

        W = self.params[name_w]
        b = self.params[name_b]
        scores, cache_affine[str(self.hidden_dims_size+1)] = affine_forward(input_, W, b) 


        # scores = out_

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores


        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

 
        loss, dout = softmax_loss(scores, y)

        # loss on last layer
        loss += 0.5*self.reg*(np.sum(np.square(self.params['W'+str(self.num_layers)])))

        ## OLD IMLEMENTATION
        #placeholder for loss
        # loss_ = 0
        # for layer_idx in range(self.hidden_dims_size):
        #     name_w = 'W'+ str(layer_idx+1)
        #     W = self.params[name_w]
            
        #     loss_ += (W * W).sum()
        # loss += 0.5 * self.reg*loss_

        # gradients flow backwards from each affine relu layers
        # for layers before output layer

        # For layer of output

        dx, dw, db = affine_backward(dout, cache_affine[str(self.hidden_dims_size+1)])

        name_w = 'W'+ str(self.hidden_dims_size+1)
        name_b = 'b'+ str(self.hidden_dims_size+1)

        # assign grads of W and b
        grads[name_w] = dw + self.reg*self.params[name_w]
        grads[name_b] = db 
        
        # set dout at first layer from the output later is dx,
        # as dx is the result from above
        #dout = dx

        for layer_idx in range(self.num_layers-1, 0, -1):

            # get index for each layer
            index_str = str(layer_idx)

            drelu = relu_backward(dx, cache_relu[index_str])

            if (self.normalization=='batchnorm'):         
                dbn, dgamma, dbeta = batchnorm_backward(drelu, cache_bn[index_str])
                dx, dw, db = affine_backward(dbn, cache_affine[index_str])


                name_gamma = 'gamma'+ index_str
                name_beta = 'beta'+ index_str

                grads[name_gamma] = dgamma
                grads[name_beta] = dbeta 
            else:     
                dx, dw, db = affine_backward(drelu, cache_affine[index_str])


            name_w = 'W'+ index_str
            name_b = 'b'+ index_str

            grads[name_w] = dw + self.reg*self.params[name_w]
            grads[name_b] = db 

            loss += 0.5 * self.reg * (np.sum(np.square(self.params[name_w])))

            #dout = dx.copy()
            
        # Don't forget to add dLoss/dW2 and dLoss/dW1


        # set all the gradients to grads
  
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

