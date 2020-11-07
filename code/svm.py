import numpy as np
from scipy.linalg import norm
import cvxopt as cvx


class SVM(object):
    '''
    SVM class
    '''

    def __init__(self, C=None):
        self.C = C
        self.__TOL = 1e-5

    def __linearKernel__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        # TODO: Implement linear kernel function
        # @x1 and @x2 are vectors
        return np.dot(x1, x2)
    #
    # def __polynomialKernel__(self, x1: np.ndarray, x2: np.ndarray, p: int) -> float:
    #     # TODO: Implement polynomial kernel function
    #     # @x1 and @x2 are vectors
    #     return ???
    #
    def __gaussianKernel__(self, x1: np.ndarray, x2: np.ndarray, sigma: float) -> float:
        # TODO: Implement gaussian kernel function
        # @x1 and @x2 are vectors
        return np.exp(-np.linalg.norm(x1-x2, axis = 1)**2 / (2*(sigma**2)))
    #
    # def __computeKernelMatrix__(self, x: np.ndarray, kernelFunction, pars) -> np.ndarray:
    #     # TODO: Implement function to compute the kernel matrix
    #     # @x is the data matrix
    #     # @kernelFunction - pass a kernel function (gauss, poly, linear) to this input
    #     # @pars - pass the possible kernel function parameter to this input
    #     return K

    def train(self, x: np.ndarray, y: np.ndarray, kernel=None, kernelpar=2) -> None:
        # TODO: Implement the remainder of the svm training function
        self.kernelpar = kernelpar

        x = x.T


        n, m = x.shape

        # Gram matrix
        K = np.zeros((n, n))




        #calculate kernel


        x = x.T

        print(x.shape)
        print(y.shape)

        y = y.reshape(1, -1) * 1.


        #H = np.outer(y.T, y) * self.__linearKernel__(x1 = x, x2 = y)

        H = np.outer(y.T, y) * np.dot(x.T, x)

        print("np.outer shape is: ", np.outer(y.T, y).shape)


        print("outer is,", H)

        print("shape of H", H.shape)

        # Converting into cvxopt format
        P = cvx.matrix(H)
        q = cvx.matrix(-np.ones((n, 1)))
        G = cvx.matrix(-np.eye(n))
        #h = cvx.matrix(np.zeros((n, 1)))
        h = cvx.matrix(np.zeros(n))
        A = cvx.matrix(y)
        b = cvx.matrix(np.zeros(1))

        print("shape of A:", y.reshape(-1, m).shape)



        # solve
        solution = cvx.solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x'])





        # we'll solve the dual
        #https://xavierbourretsicotte.github.io/SVM_implementation.html

        #commented out for solving the linear SVM, only needed for the Kernel SVM implementation (2)
        # # obtain the kernel
        # if kernel == 'linear':
        #     # TODO: Compute the kernel matrix for the non-linear SVM with a linear kernel
        #     print('Fitting SVM with linear kernel')
        #     K = ???
        #     self.kernel = self.__linearKernel__
        # elif kernel == 'poly':
        #     # TODO: Compute the kernel matrix for the non-linear SVM with a polynomial kernel
        #     print('Fitting SVM with Polynomial kernel, order: {}'.format(kernelpar))
        #     K = ???
        # elif kernel == 'rbf':
        #     # TODO: Compute the kernel matrix for the non-linear SVM with an RBF kernel
        #     print('Fitting SVM with RBF kernel, sigma: {}'.format(kernelpar))
        #     K = ???
        # else:
        #     print('Fitting linear SVM')
        #     # TODO: Compute the kernel matrix for the linear SVM
        #     K = ???
        #
        if self.C is None:
            G = None
            h = None
        else:
            print("Using Slack variables")
            G = None
            h = None


        # TODO: Compute below values according to the lecture slides
        self.lambdas = None # Only save > 0
        self.sv = None # List of support vectors
        self.sv_labels = None # List of labels for the support vectors (-1 or 1 for each support vector)
        if kernel is None:
          self.w = None # SVM weights used in the linear SVM
          # Use the mean of all support vectors for stability when computing the bias (w_0)
          self.bias = None # Bias
        else:
          self.w = None
          # Use the mean of all support vectors for stability when computing the bias (w_0).
          # In the kernel case, remember to compute the inner product with the chosen kernel function.
          self.bias = None # Bias

        # TODO: Implement the KKT check
        self.__check__()

    def __check__(self) -> None:
        # Checking implementation according to KKT2 (Linear_classifiers slide 46)
        kkt2_check = None
        assert kkt2_check < self.__TOL, 'SVM check failed - KKT2 condition not satisfied'

    def classifyLinear(self, x: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained linear SVM - access the SVM parameters through self.
        :param x: Data to be classified
        :return: List of classification values (-1.0 or 1.0)
        '''
        # TODO: Implement
        #return ???
        pass

    def printLinearClassificationError(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls classifyLinear and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement
        print("Total error: {:.2f}%".format(result))

    def classifyKernel(self, x: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained kernel SVM - use self.kernel and self.kernelpar to access the kernel function and parameter
        :param x: Data to be classified
        :return: List of classification values (-1.0 or 1.0)
        '''
        # TODO: Implement
        #return ???
        pass

    def printKernelClassificationError(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls classifyKernel and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement
        print("Total error: {:.2f}%".format(result))