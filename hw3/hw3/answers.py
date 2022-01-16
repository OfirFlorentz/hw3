r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.01, 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 0.015, 0.005, 0.001, 0.004

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0.09, 0.002
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""

We can see that without dropout, we got nearly 100% accuracy on the training set, which implies on overfitting,
while with dropout we got less accuracy on the training set and higher accuracy on the test set.
Which is in correlation to our expectation because we learned that adding dropout is affected by reducing the overfitting.

Also we can see difference between the 0.4 dropout and the 0.8 dropout whereas on the 0.8 we got less accuracy on the
training compared to the 0.4 dropout set which implies on reducing the overfitting, but on the 0.4 dropout we got
much higher accuracy on the test set, which might be because on the 0.8 we lose too much data and therefore reducing
the tests set accuracy.

"""

part2_q2 = r"""

Yes. As we can see from the graphs, from epochs 25-30 on the test set, we get increasing loss in the 0.4 dropout while
on the same epoch we get increasing accuracy.
That can explained by the fact that in order for the accuracy to change it needs to cross certain threshold, while the
loss can also be affected by minor changes, causing the effect we just described. 


"""

part2_q3 = r"""

1.
    Back propagation is a way of calculating gradients for the weights of the network,
    whereas Gradient descent is algorithm for reaching local minima of the loss function.
2.
    In Gradient-Descent we use the entire training set (Epoch) to calculate the gradient and then adjust the weights
     accordingly.
    In Stochastic gradient descent we adjust the weights after every smaller amount of samples,
     depend on the stochastic algorithm, and calculate the gradient according to those samples.
3.
    A. Normal Gradient descent is more likely to stuck on a local minima, whereas SGD is more likely  to avoid local minima
        resulting in better results.
    B. Normal Gradient descent is slower because we first need to predict the entire training set and then calculate
        the gradient according to that. In SGD the calculation time is much faster because we calculate the gradient on
         smaller amount of samples. 
    
"""
# ==============

# ==============
# Part 3 answers


def part3_rnn_hyperparams():
    hypers = dict(
        batch_size=250,
        seq_len=50,
        h_dim=1000,
        n_layers=2,
        dropout=0.2,
        learn_rate=0.001,
        lr_sched_factor=0.35,
        lr_sched_patience=2,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    # ========================
    return hypers


def part3_generation_params():
    start_seq = "Long time ago I woke from a deep sleep"
    temperature = 0.4
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    # ========================
    return start_seq, temperature


part3_q1 = r"""

Splitting the corpus into sequences improves the training time significantly, training on the whole data set will
    take too long to train. Moreover training on the whole set can result in vanishing gradients, resulting in
    non-trainable network, and splitting the set solves that.     

"""

part3_q2 = r"""

Because the generated text includes the history so after some iterations, the generated text will me longer the the
 sequence length because of the history includes in it.

"""

part3_q3 = r"""

Because in our problem the order of the text matters, and we rely on it in our predictions.
 Shuffling the batches will result in non logical sentences and will harm the results.


"""

part3_q4 = r"""

1. With a low temperature, we reduce the chance of low score character to be chosen.
    We use lower temperatures here to allow the model to pick a character that is based on what the model has
    already learnt.

2. If the temperature is very high, also the low scored character will get decent probability to be chosen, resulting
    in big variance in the prediction regardless of the scores.
    
3. Low temperature will cause in amplifying the probability of the higher scored characters to be chose, resulting in 
    almost deterministic action, predicting the same character every time.


"""
# ==============
