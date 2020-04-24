    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
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

        fc_cache = {}
        relu_cache = {}
        bn_cache = {}
        dropout_cache = {}
        batch_size = X.shape[0]

        X = np.reshape(X, [batch_size, -1])  # Flatten our input images.

        # Do as many Affine-Relu forward passes as required (num_layers - 1).
        # Apply batch norm and dropout as required.
        for i in range(self.num_layers-1):

            fc_act, fc_cache[str(i+1)] = affine_forward(X, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
            if self.normalization == 'batchnorm':
                bn_act, bn_cache[str(i+1)] = batchnorm_forward(fc_act, self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i])
                relu_act, relu_cache[str(i+1)] = relu_forward(bn_act)
            else:
                relu_act, relu_cache[str(i+1)] = relu_forward(fc_act)
            if self.use_dropout:
                relu_act, dropout_cache[str(i+1)] = dropout_forward(relu_act, self.dropout_param)

            X = relu_act.copy()  # Result of one pass through the affine-relu block.

        # Final output layer is FC layer with no relu.
        scores, final_cache = affine_forward(X, self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])

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
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        # Calculate score loss and add reg. loss for last FC layer.
        loss, dsoft = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.sum(np.square(self.params['W'+str(self.num_layers)])))

        # Backprop dsoft to the last FC layer to calculate gradients.
        dx_last, dw_last, db_last = affine_backward(dsoft, final_cache)

        # Store gradients of the last FC layer
        grads['W'+str(self.num_layers)] = dw_last + self.reg*self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = db_last

        # Iteratively backprop through each Relu & FC layer to calculate gradients.
        # Go through batchnorm and dropout layers if needed.
        for i in range(self.num_layers-1, 0, -1):

            if self.use_dropout:
                dx_last = dropout_backward(dx_last, dropout_cache[str(i)])

            drelu = relu_backward(dx_last, relu_cache[str(i)])

            if self.normalization == 'batchnorm':
                dbatchnorm, dgamma, dbeta = batchnorm_backward(drelu, bn_cache[str(i)])
                dx_last, dw_last, db_last = affine_backward(dbatchnorm, fc_cache[str(i)])
                grads['beta' + str(i)] = dbeta
                grads['gamma' + str(i)] = dgamma
            else:
                dx_last, dw_last, db_last = affine_backward(drelu, fc_cache[str(i)])

            # Store gradients.
            grads['W' + str(i)] = dw_last + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db_last

            # Add reg. loss for each other FC layer.
            loss += 0.5 * self.reg * (np.sum(np.square(self.params['W' + str(i)])))


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
