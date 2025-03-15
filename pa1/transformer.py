import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28

def transformer(X: ad.Node, nodes: List[ad.Node], 
                      model_dim: int, seq_length: int, eps, batch_size, num_classes) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """

    """TODO: Your code here"""
    W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = nodes

    # (batch_size, seq_length, model_dim) @ (model_dim, model_dim) -> (batch_size, seq_length, model_dim)
    q = ad.matmul(X, W_Q)  
    k = ad.matmul(X, W_K) 
    v = ad.matmul(X, W_V)
    # attn
    attn = ad.matmul(q, ad.transpose(k, 1, 2))
    attn = ad.div_by_const(attn, np.sqrt(model_dim))  # Scale by sqrt(model_dim)
    attn = ad.softmax(attn)
    attn = ad.matmul(attn, v)
    attn = ad.matmul(attn, W_O)
    X = ad.layernorm(attn, normalized_shape=[model_dim], eps=eps)

    # FFN
    X = ad.add(ad.matmul(X, W_1), b_1)
    X = ad.relu(X)
    X = ad.layernorm(X, normalized_shape=[model_dim], eps=eps)

    X = ad.sum_op(X, dim=1, keepdim=False)
    X = ad.div_by_const(X, seq_length)

    logits = ad.add(ad.matmul(X, W_2), b_2)

    return logits



def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    """TODO: Your code here"""
    softmax_Z = ad.softmax(Z, dim=-1)
    log_softmax_Z = ad.log(softmax_Z)
    loss = ad.sum_op(ad.mul(y_one_hot, log_softmax_Z), dim=1)

    # loss = ad.sum_op(loss, dim=0) # ExpandAsOp3d has bug
    avg_loss = ad.div_by_const(loss, batch_size)
    avg_loss = ad.mul_by_const(avg_loss, -1)
    
    return avg_loss



def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    """TODO: Your code here"""
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size> num_examples:continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]
        
        # Compute forward and backward passes
        # TODO: Your code here
        logits, loss, *grad = f_run_model(model_weights, X_batch, y_batch)
        
        # Update weights and biases
        # TODO: Your code here
        # Hint: You can update the tensor using something like below:
        # W_Q -= lr * grad_W_Q.sum(dim=0)

        # Accumulate the loss
        # TODO: Your code here
        for w, g in (zip(model_weights[:5], grad[:5])):
            w -= lr * g.sum(dim=0)
        model_weights[5] -= lr * grad[5].sum(dim=0)
        model_weights[6] -= lr * grad[6].sum(dim=(0, 1))
        model_weights[7] -= lr * grad[7].sum(dim=(0, 1))

    # Compute the average loss
    total_loss += loss

    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)

    # TODO: Your code here
    # You should return the list of parameters and the loss
    return model_weights, average_loss

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # TODO: Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10 #
    model_dim = 128 #
    eps = 1e-5 

    # - Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # TODO: Define the forward graph.
    X = ad.Variable(name="X")
    y_groundtruth = ad.Variable(name="y")
    W_Q = ad.Variable(name="W_Q")
    W_K = ad.Variable(name="W_K")
    W_V = ad.Variable(name="W_V")
    W_O = ad.Variable(name="W_O")
    W_1 = ad.Variable(name="W_1")
    W_2 = ad.Variable(name="W_2")
    b_1 = ad.Variable(name="b_1")
    b_2 = ad.Variable(name="b_2")
    
    nodes = [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

    y_predict: ad.Node = transformer(X, nodes, model_dim, seq_length, eps, batch_size, num_classes) # TODO: The output of the forward pass
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    # TODO: Construct the backward graph.
    

    # TODO: Create the evaluator.
    grads: List[ad.Node] = ad.gradients(loss, nodes) # TODO: Define the gradient nodes here
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(model_weights, X_batch, y_batch):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        W_Q_val, W_K_val, W_V_val, W_O_val, W_1_val, W_2_val, b_1_val, b_2_val = model_weights
        result = evaluator.run(
            input_values={
                # TODO: Fill in the mapping from variable to tensor
                X: X_batch,
                y_groundtruth: y_batch ,
                W_Q: W_Q_val,
                W_K: W_K_val,
                W_V: W_V_val,
                W_O: W_O_val, 
                W_1: W_1_val, 
                W_2: W_2_val, 
                b_1: b_1_val,
                b_2: b_2_val
            }
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        total_loss = 0.0
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size> num_examples:continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            W_Q_val, W_K_val, W_V_val, W_O_val, W_1_val, W_2_val, b_1_val, b_2_val = model_weights
            logits = test_evaluator.run({
                # TODO: Fill in the mapping from variable to tensor
                X: X_batch,
                W_Q: W_Q_val,
                W_K: W_K_val,
                W_V: W_V_val,
                W_O: W_O_val,
                W_1: W_1_val,
                W_2: W_2_val,
                b_1: b_1_val,
                b_2: b_2_val
            })
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    model_weights: List[torch.Tensor] = [
                    torch.tensor(W_Q_val), torch.tensor(W_K_val),
                    torch.tensor(W_V_val), torch.tensor(W_O_val),
                    torch.tensor(W_1_val), torch.tensor(W_2_val),
                    torch.tensor(b_1_val), torch.tensor(b_2_val)
    ] # TODO: Initialize the model weights here
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
