import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
import datetime
import sparseoperations
from copy import deepcopy
from dataManager import DataManager

def backpropagation_updates_Numpy(a, delta, rows, cols, out, layer):
    for i in range (out.shape[0]):
        s = 0
        for j in range(a.shape[0]):
            s+=a[j,rows[i]]*delta[j, cols[i]]
        out[i]=s/a.shape[0]

def compute_hebbian_factor(activations_avg, activations, activations_next, rows, cols, out, layer):
    for i in range(rows.shape[0]):
        delta_w = 0
        for j in range (activations.shape[0]):
            delta_w += activations[j, rows[i]] * (activations_next[j, cols[i]] - activations_avg[cols[i]])
        out[i] = delta_w / activations.shape[0]

def createSparseWeights(epsilon,noRows,noCols):
    # generate an Erdos Renyi sparse weights mask
    weights=lil_matrix((noRows, noCols))
    for i in range(epsilon * (noRows + noCols)):
        weights[np.random.randint(0,noRows),np.random.randint(0,noCols)]=np.float64(np.random.randn()/10)
    print("Creating sparse matrix of shape ", noRows, ", ", noCols)
    print ("Create sparse matrix with ",weights.getnnz()," connections and ",(weights.getnnz()/(noRows * noCols))*100,"% density level")
    weights=weights.tocsr()
    return weights


def array_intersect(A, B, arrays_are_unique = False):
    # added by Amarsagar Reddy Ramapuram Matavalam (amar@iastate.edu)
    # this are for array intersection
    # inspired by https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [A.dtype]}
    return np.in1d(A.view(dtype), B.view(dtype), assume_unique=arrays_are_unique)  # boolean return

class Relu:
    @staticmethod
    def activation(z):
        out = np.zeros(z.shape)
        out[z > 0] = z[z > 0]
        return out

    @staticmethod
    def prime(z):
        out = np.zeros(z.shape)
        out[z > 0] = 1
        return out

class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))
    
class CrossEntropy:
    def __init__(self, activation_fn=None):
        """

        :param activation_fn: Class object of the activation function.
        """
        if activation_fn:
            self.activation_fn = activation_fn
        else:
            self.activation_fn = NoActivation

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.sum(np.nan_to_num(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)))

    def delta(self, y_true, y_pred):
        """MSE
        Back propagation error delta
        :return: (array)
        """
        return y_pred - y_true

class MSE:
    def __init__(self, activation_fn=None):
        """

        :param activation_fn: Class object of the activation function.
        """
        if activation_fn:
            self.activation_fn = activation_fn
        else:
            self.activation_fn = NoActivation

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_pred - y_true)**2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)

class NoActivation:
    """
    This is a plugin function for no activation.

    f(x) = x * 1
    """
    @staticmethod
    def activation(z):
        """
        :param z: (array) w(x) + b
        :return: z (array)
        """
        print("Using NoActivation activation")
        return z

    @staticmethod
    def prime(z):
        """
        The prime of z * 1 = 1
        :param z: (array)
        :return: z': (array)
        """
        print("Using NoActivation prime")
        return np.ones_like(z)

class SET_MLP:
    def __init__(self, dimensions, activations, epsilon=20):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.

        Example of three hidden layer with
        - 3312 input features
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 5 output classes


        layers -->    [1,        2,     3,     4,     5]
        ----------------------------------------

        dimensions =  (3312,     3000,  3000,  3000,  5)
        activations = (          Relu,  Relu,  Relu,  Sigmoid)
        """
        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None
        self.momentum=None
        self.weight_decay = None
        self.epsilon = epsilon  # control the sparsity level as discussed in the paper
        self.zeta = None  # the fraction of the weights removed
        self.hebbian_factor = None
        self.dimensions=dimensions

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw={}
        self.pdd={}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}
        for i in range(len(dimensions) - 1):
            self.w[i + 1] = createSparseWeights(self.epsilon, dimensions[i], dimensions[i + 1])#create sparse weight matrices
            self.b[i + 1] = np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]

    def _feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """
        z = {}
        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.

        for i in range(1, self.n_layers):
            # current layer = i
            # activation layer = i + 1
            z[i + 1] = a[i]@self.w[i] + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])

        return z, a

    def _back_prop(self, z, a, y_true):
        """
        The input dicts keys represent the layers of the net.

        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              4: f(w3(a3) + b3)
              5: f(w4(a4) + b4)
              }

        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw=coo_matrix(self.w[self.n_layers-1])

        # compute backpropagation updates
        sparseoperations.backpropagation_updates_Cython(a[self.n_layers - 1],delta,dw.row,dw.col,dw.data)
        # If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
        #backpropagation_updates_Numpy(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data, self.n_layers-1)

        update_params = {
            self.n_layers - 1: (dw.tocsr(), delta)
        }

        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):
            delta = (delta@self.w[i].transpose()) * self.activations[i].prime(z[i])
            dw = coo_matrix(self.w[i - 1])
            
            #compute backpropagation updates
            sparseoperations.backpropagation_updates_Cython(a[i - 1], delta, dw.row, dw.col, dw.data)
            #If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
            #backpropagation_updates_Numpy(a[i - 1], delta, dw.row, dw.col, dw.data, i-1)

            update_params[i - 1] = (dw.tocsr(), delta)

        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])
        return update_params

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.

        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """
        self.w[index] -= self.learning_rate * dw
        self.b[index] -= self.learning_rate * np.mean(delta, 0)

    def fit(self, x, y_true, x_test,y_test,loss, epochs, batch_size, learning_rate=1e-3, zeta=0.3, hebbian_factor = 0.0, hebbian_decay_factor = 0.0, testing=True, save_filename=""):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        :param momentum: (flt)
        :param weight_decay: (flt)
        :param zeta: (flt) #control the fraction of weights removed
        :param hebbian_factor (flt) # strength of Hebbian learning factor in weight updates
        :return (array) A 2D array of metrics (epochs, 3).
        """
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
        
        # Initiate the loss object with the final activation function
        self.loss = loss(self.activations[self.n_layers])
        self.learning_rate = learning_rate
        self.zeta = zeta
        self.hebbian_factor = hebbian_factor
        self.hebbian_decay_factor = hebbian_decay_factor

        maximum_accuracy=0

        metrics=np.zeros((epochs,3))

        for i in range(epochs):
            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_=x[seed]
            y_=y_true[seed]

            #training
            t1 = datetime.datetime.now()
            losstrain=0

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                z, a = self._feed_forward(x_[k:l])                

                losstrain+=self.loss.loss(y_[k:l], a[self.n_layers])

                z_copy = deepcopy(z)
                dw_backprop = self._back_prop(z_copy, a, y_[k:l])

                # Compute and apply Hebbian learning factor
                if self.hebbian_factor > 0.0:
                    
                    # Normalise activations to range [0,1] over each batch
                    a_norm = {}
                    for key, activations in a.items():
                        a_norm[key] = activations - np.min(activations, 0)
                        a_norm[key] /= (np.max(a_norm[key], 0) + 1e-4)

                    # Compute average activations
                    a_avg = {}
                    for key, activations in a_norm.items():
                        a_avg[key] = np.mean(activations, 0)

                    for layer in range(1, self.n_layers):
                        hebbian_weight_update = np.empty(self.w[layer].data.shape)

                        w_coo = self.w[layer].tocoo()

                        #Uncomment line below if there are issues with the Cython implementation- note that it will be slower
                        #compute_hebbian_factor(a_avg[layer + 1], a[layer], a[layer+1], w_coo.row, w_coo.col, hebbian_weight_update, layer)
                        
                        # Compute hebbian updates
                        sparseoperations.compute_hebbian_factor(a_avg[layer + 1], a_norm[layer], a_norm[layer+1], w_coo.row, w_coo.col, hebbian_weight_update)
                        hebbian_weight_update *= self.hebbian_factor

                        self.w[layer].data += hebbian_weight_update

            # Decay Hebbian factor each epoch
            self.hebbian_factor *= 1 - self.hebbian_decay_factor

            t2 = datetime.datetime.now()
            metrics[i, 0]=losstrain / (x.shape[0] // batch_size)
            print ("\nSET-MLP Epoch ",i)
            print ("Training time: ",t2-t1,"; Loss train: ",losstrain / (x.shape[0] // batch_size))

            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            if (testing):
                t3 = datetime.datetime.now()
                accuracy,activations=self.predict(x_test,y_test,batch_size)
                t4 = datetime.datetime.now()
                maximum_accuracy=max(maximum_accuracy,accuracy)
                losstest=self.loss.loss(y_test[:activations.shape[0]], activations)
                metrics[i, 1] = accuracy
                metrics[i, 2] = losstest

                accuracy_train,_=self.predict(X_train,Y_train,batch_size)
                print("accuracy train = ", accuracy_train)
                print("Testing time: ", t4 - t3, "; Loss test: ", losstest,"; Accuracy: ", accuracy,"; Maximum accuracy: ", maximum_accuracy)

            t5 = datetime.datetime.now()

            # Remove weights. Add new weights unless it is the last epoch
            self.weightsEvolution(add_new_weights = i != epochs-1)

            t6 = datetime.datetime.now()
            print("Weights evolution time ", t6 - t5)
        
            #save performance metrics values in a file
            if (save_filename!=""):
                np.savetxt(save_filename,metrics)

        return metrics  
    
    def weightsEvolution(self, add_new_weights = True):
        for i in range(1,self.n_layers):
            w_coo = self.w[i].tocoo()
            data_w= w_coo.data
            rows_w = w_coo.row
            cols_w = w_coo.col
            n_weights = len(data_w)

            # Find the indices of the zeta % smallest weights
            to_remove = np.argpartition(abs(data_w), int(n_weights * self.zeta + 1))[:int(n_weights * self.zeta + 1)]
            data_w = np.delete(data_w, to_remove)
            rows_w = np.delete(rows_w, to_remove)
            cols_w = np.delete(cols_w, to_remove)
            
            if add_new_weights:
                to_add = n_weights - len(data_w)
                while (to_add>0):
                    ik = np.random.randint(0, self.dimensions[i - 1], size=to_add, dtype='int32')
                    jk = np.random.randint(0, self.dimensions[i], size=to_add, dtype='int32')
                
                    w_coo_to_add = np.stack((ik,jk), axis=-1)
                    w_coo_to_add = np.unique(w_coo_to_add, axis=0) # removing duplicates in new rows&cols
                    old_coordinates =np.stack((rows_w, cols_w), axis=-1)

                    uniqueFlag= ~array_intersect(w_coo_to_add, old_coordinates, arrays_are_unique=True) # careful about order & tilda, unique flag speeds up computation

                    rows_w = np.append(rows_w, w_coo_to_add[uniqueFlag,0])
                    cols_w = np.append(cols_w, w_coo_to_add[uniqueFlag,1])

                    to_add = n_weights - np.size(rows_w)

                data_w = np.append(data_w, np.random.randn(n_weights - len(data_w))/10)
            self.w[i]=csr_matrix((data_w , (rows_w , cols_w)), shape = (self.dimensions[i-1],self.dimensions[i]))

    def predict(self, x_test,y_test,batch_size=1):
        """
        :param x_test: (array) Test input
        :param y_test: (array) Correct test output
        :param batch_size:
        :return: (flt) Classification accuracy
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        activations = np.zeros(((x_test.shape[0] // batch_size) * batch_size, y_test.shape[1]))
        for j in range(x_test.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            _, a_test = self._feed_forward(x_test[k:l])
            activations[k:l] = a_test[self.n_layers]
        correctClassification = 0
        for j in range(activations.shape[0]):
            if (np.argmax(activations[j]) == np.argmax(y_test[j])):
                correctClassification += 1
        accuracy= correctClassification/activations.shape[0]
        return accuracy, activations

if __name__ == "__main__":
    # Comment this if you would like to use the full power of randomization. I use it to have repeatable results.
    np.random.seed(0)

    (X_train, Y_train), (X_test, Y_test) = DataManager.get_lung_data()

    hebbian_factor = 0.01 # Hyperparameter to tune strength of Hebbian factor
    hebbian_decay_factor = 0.02 # Hyperparameter to tune how quickly Hebbian factor decreases

    start = datetime.datetime.now()
    # create SET-MLP
    set_mlp = SET_MLP((X_train.shape[1], 1000, 1000, 1000, Y_train.shape[1]), (Relu, Relu, Relu, Sigmoid), epsilon=20)

    # train SET-MLP
    set_mlp.fit(X_train, Y_train, X_test, Y_test, loss=MSE, epochs=50, batch_size=20, learning_rate=0.02, hebbian_factor = hebbian_factor, hebbian_decay_factor= hebbian_decay_factor, zeta=0.3, testing=True, save_filename= '')

    end = datetime.datetime.now()
    print("Total training time = ", end-start)

    # test SET-MLP
    accuracy, activations =set_mlp.predict(X_test,Y_test,batch_size=1)
    print ("\nAccuracy of the last epoch on the testing data: ", accuracy)