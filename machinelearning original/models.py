import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        val = nn.as_scalar(self.run(x))
        return 1 if val >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        converged = True
        for x, label in dataset.iterate_once(1):
            y_pred = self.get_prediction(x)
            if nn.as_scalar(label) != y_pred:
                self.w.update(nn.Constant(nn.as_scalar(label)*x.data), 1)
                converged = False
        if converged:
            return
        self.train(dataset)
                
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_size = 100 # [100, 500]
        self.batch_size = 1 # [1, 128]
        self.learning_rate = 0.01 # [0.0001, 0.01]
        # parameters must be updated as we train 
        self.W1 = nn.Parameter(1, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.W2 = nn.Parameter(self.hidden_size, 1)
        self.b2 = nn.Parameter(1, 1)
        self.helpers = Helpers(self.learning_rate, nn.SquareLoss, self)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = nn.AddBias(nn.Linear(x, self.W1), self.b1)
        x = nn.Linear(nn.ReLU(x), self.W2)
        return nn.AddBias(x, self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return self.helpers.loss(x, y)

    def train(self, dataset, num_epochs=1000, target_loss=0.02):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        if num_epochs <= 0:
            return
        total_loss = 0
        for x, label in dataset.iterate_once(self.batch_size):
            total_loss += self.helpers.update_weights(x, label)
        avg_loss = total_loss / dataset.x.shape[0]
        if avg_loss < target_loss:
            return
        self.train(dataset, num_epochs-1)
                 

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_size = 200
        self.batch_size = 48
        self.learning_rate = 0.2
        self.W1 = nn.Parameter(784, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.W2 = nn.Parameter(self.hidden_size, 10)
        self.b2 = nn.Parameter(1, 10)
        self.helpers = Helpers(self.learning_rate, nn.SoftmaxLoss, self)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        x = nn.AddBias(nn.Linear(x, self.W1), self.b1)
        x = nn.Linear(nn.ReLU(x), self.W2)
        return nn.AddBias(x, self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return self.helpers.loss(x, y)

    def train(self, dataset, num_epochs=20, target_accuracy=0.98):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        if num_epochs <= 0:
            return
        if dataset.get_validation_accuracy() >= target_accuracy:
            return
        for x, label in dataset.iterate_once(self.batch_size):
            self.helpers.update_weights(x, label)
        self.train(dataset, num_epochs-1)


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_size = 100 # [100, 500]
        self.batch_size = 32 # [1, 128]
        self.learning_rate = 0.01 # [0.0001, 0.01]
        self.W1 = nn.Parameter(self.num_chars, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.W2 = nn.Parameter(self.hidden_size, len(self.languages))
        self.b2 = nn.Parameter(1, len(self.languages))
        self.result = nn.Parameter(5, self.hidden_size)
        self.helpers = Helpers(self.learning_rate, nn.SoftmaxLoss, self)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        return self.run_helper(xs, 1, self.initialize(xs[0]))
    
    def run_helper(self, xs, i, x):
        if i >= len(xs):
            return x
        x = nn.AddBias(nn.Add(nn.Linear(xs[i], self.W1), nn.Linear(x, self.result)), self.b1)
        x = nn.AddBias(nn.Linear(nn.ReLU(x), self.W2), self.b2)
        return self.run_helper(xs, i+1, x)

    def initialize(self, x):
        x = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        x = nn.AddBias(nn.Linear(x, self.W2), self.b2)
        return x
            
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return self.helpers.loss(xs, y)

    def train(self, dataset, num_epochs=110, target_accuracy=0.87):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        if num_epochs <= 0:
            return
        if dataset.get_validation_accuracy() >= target_accuracy:
            return
        for x, label in dataset.iterate_once(self.batch_size):
            self.helpers.update_weights(x, label, True)
        self.train(dataset, num_epochs-1) 
        
# General functions every class uses:
class Helpers():
    def __init__(self, learning_rate, loss_function, model):
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.model = model
        
    def get_gradients(self, loss, weight_result=False):
        if weight_result:
            return  nn.gradients(loss, [self.model.W1, self.model.b1, self.model.W2, self.model.b2, self.model.result])
        return nn.gradients(loss, [self.model.W1, self.model.b1, self.model.W2, self.model.b2])
        
    def update_weights(self, x, label, weight_result=False):
        loss = self.model.get_loss(x, label)
        gradients = self.get_gradients(loss, weight_result)
        self.model.W1.update(gradients[0], -self.learning_rate)
        self.model.b1.update(gradients[1], -self.learning_rate)
        self.model.W2.update(gradients[2], -self.learning_rate)
        self.model.b2.update(gradients[3], -self.learning_rate)
        if weight_result: 
            self.model.result.update(gradients[4], -self.learning_rate)
        return nn.as_scalar(loss)
        
    def loss(self, x, y):
        y_pred = self.model.run(x)
        return self.loss_function(y_pred, y)
            
