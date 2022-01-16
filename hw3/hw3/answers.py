r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


part1_q1 = r"""
1.a -
We have batch_size * dim_in parameters that go in and  batch_size * dim_out parameters that go out.
therefore the shape of the tensor is batch_size * dim_in * batch_size * dim_out. (1024 * 64 * 512 * 64).

1. b - Yes. We have 64 different input, each input is i.i.d.
therefore the derivative of the input parameter  0i to the output parameter 0j will be 0.  
So only 1024*512*64 parameters will be non zero and the rest ( 1024*512*64 * 63) will be zero.

1.c - No. We know that ($\delta\mat{X}$) is equal to  $\delta\mat{Y}*\pderiv{Y}{\mat{X}}$ but in fact 
$\pderiv{Y}{\mat{X}}$ = W that we already have. 

2. a - The tensor shape will be the shape of W * shape of X. e.g 1024 * 64 * 512 * 1064.

2. b - Yes, from the same reason in 1.b each example is i.i.d and therefore the i row in the input matrix row
 will not influence the j row of the output

2.c - Similar to 1.c We can calculate ($\delta\mat{W}$) is equal to  $\delta\mat{Y}*\pderiv{Y}{\mat{W}}= 
dot(\delta\mat{Y}, X)$  

"""
part1_q2 = r"""
It is theoretically possible to train neural net without back propagation. for ex we can make a a random search 
in the weight space and try to look for a good result in the loss function. there several technique to do it.
(https://towardsdatascience.com/the-fascinating-no-gradient-approach-to-neural-net-optimization-abb287f88c97)

But it seems that the GD method do achieve better result then any other method, especially in a complex problems. 

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
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 0.015, 0.005, 0.0005, 0.004

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
**Your answer:**

We can see that without dropout, we got nearly 100% accuracy on the training set, which implies on overfitting,
while with dropout we got less accuracy on the training set and higher accuracy on the test set.
Which is in correlation to our expectation because we learned that adding dropout is affected by reducing the overfitting.

Also we can see difference between the 0.4 dropout and the 0.8 dropout whereas on the 0.8 we got less accuracy on the
training compared to the 0.4 dropout set which implies on reducing the overfitting, but on the 0.4 dropout we got
much higher accuracy on the test set, which might be because on the 0.8 we lose too much data and therefore reducing
the tests set accuracy.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**

Yes. As we can see from the graphs, from epochs 25-30 on the test set, we get increasing loss in the 0.4 dropout while
on the same epoch we get increasing accuracy.
That can explained by the fact that in order for the accuracy to change it needs to cross certain threshold, while the
loss can also be affected by minor changes, causing the effect we just described. 


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**

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
    

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
**Your answer:**

Splitting the corpus into sequences improves the training time significantly, training on the whole data set will
    take too long to train. Moreover training on the whole set can result in vanishing gradients, resulting in
    non-trainable network, and splitting the set solves that.     

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**

Because the generated text includes the history so after some iterations, the generated text will me longer the the
 sequence length because of the history includes in it.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**

Because in our problem the order of the text matters, and we rely on it in our predictions.
 Shuffling the batches will result in non logical sentences and will harm the results.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:**

1. With a low temperature, we reduce the chance of low score character to be chosen.
    We use lower temperatures here to allow the model to pick a character that is based on what the model has
    already learnt.

2. If the temperature is very high, also the low scored character will get decent probability to be chosen, resulting
    in big variance in the prediction regardless of the scores.
    
3. Low temperature will cause in amplifying the probability of the higher scored characters to be chose, resulting in 
    almost deterministic action, predicting the same character every time.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
