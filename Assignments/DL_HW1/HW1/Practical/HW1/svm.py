import numpy as np


class SVM:
    def __init__(self, n_features: int, n_classes: int, std: float):
        """
        n_features: number of features in (or the dimension of) each instance
        n_classes: number of classes in the classification task
        std: standard deviation used in the initialization of the weights of svm
        """
        self.n_features, self.n_classes = n_features, n_classes
        self.cache = None
        ################################################################################
        # TODO: Initialize the weights of svm using random normal distribution with    #
        # standard deviation equals to std.                                            #
        ################################################################################
        pass
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################

    def loss(self, X: np.ndarray, y: np.ndarray, reg_coeff: float):
        """
        X: training instances as a 2d-array with shape (num_train, n_features)
        y: labels corresponsing to the given training instances as a 1d-array with shape (num_train,)
        reg_coeff: L2-regularization coefficient
        """
        loss = 0.0
        ################################################################################
        # TODO: Compute the hinge loss specified in the notebook and save it in the    #                                                   # loss variable.                                                               #
        # NOTE: YOU ARE NOT ALLOWED TO USE FOR LOOPS!                                  #
        # Don't forget L2-regularization term in your implementation!                  #
        # You might need some values computed here when you will update the weights.   #
        # save them in self.cache attribute and use them in update_weights method.     #
        ################################################################################
        pass
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        return loss
        
    def update_weights(self, learning_rate: float):
        """
        Updates the weights of the svm using the gradient of computed loss w.r.t. the weights. 
        learning_rate: learning rate that will be used in gradient descent to update the weights
        """
        grad_W = None
        ################################################################################
        # TODO: Compute the gradient of loss computed above w.r.t the svm weights.     # 
        # the gradient will be used for updating the weights.                          #
        # NOTE: YOU ARE NOT ALLOWED TO USE FOR LOOPS!                                  #
        # Don't forget L2-regularization term in your implementation!                  #
        # You can use the values saved in cache attribute previously during the        #
        # computation of loss here.                                                    # 
        ################################################################################
        pass
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        self.W -= learning_rate * grad_W
        return grad_W
        
    def predict(self, X):
        """
        X: Numpy 2d-array of instances
        """
        y_pred = None
        ################################################################################
        # TODO: predict the labels for the instances in X and save them in y_pred.     #
        # Hint: You might want to use np.argmax.                                       #
        ################################################################################
        pass
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        return y_pred