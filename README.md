# Training an LSTM network on the Penn Tree Bank (PTB) dataset
---

## Introduction

Long Short-Term Memory (LSTM) networks were first proposed by Sepp Hochreiter and [Jürgen Schmidhuber][1] in 1997 for modeling sequence data. [Christopher Olah][2] has nicely illustrated how they work. The fifth course in the [deep learning specialization][3] on Coursera teaches recurrent neural networks (RNN), of which the LSTM is a variant, in detail, and explains many interesting applications. For a succinct summary of the mathematics of these models, see, for example, [Stanford cs231n lecture 10][4] or [Greff, et al. (2016)][5].

This is a series of illustrative examples of training an LSTM network. In these examples, an LSTM network is trained on the Penn Tree Bank (PTB) dataset to replicate some previously published work. The PTB dataset is an English corpus available from Tomáš Mikolov's web [page][6], and used by many researchers in language modeling experiments. It contains 929K training words, 73K validation words, and 82K test words. It has 10K words in its vocabulary. Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals used this dataset in their ICLR 2015 [paper][7] where they showed that the correct place to implement dropout regularization in an RNN is in the connections between layers and not between time steps. To demonstrate the effectiveness of their regularization strategy, they reported word-level perplexities on the PTB dataset with three different networks: a "small" non-regularized LSTM, a "medium" regularized LSTM, and a "large" regularized LSTM. It is their "small" non-regularized LSTM model that is replicated in these examples.  

Part [I](lstm_np.ipynb) of this series (i.e. this notebook) presents an object-oriented design of the non-regularized LSTM network implemented in pure [Python][8]/[Numpy][9]. Equations are coded up from scratch to carry out the computations without dependencies on extraneous frameworks or libraries. This is a minimalist implementation, partly inspired by Andrej Karpathy's [minimalist character-level language model][14]. The program is executed on a CPU. 

Part [II](lstm_tfe.ipynb) shows how the same model can be easily implemented using [TensorFlow][10], the open-source framework originally developed by researchers and engineers from the Google Brain team within Google’s AI organization. The model is programmed in TensorFlow's "[eager execution][11]" imperative programming environment that evaluates operations immediately without building dataflow graphs. This is akin to regular Python programming following Python control flow. The program is executed in [Colaboratory][12] with GPU acceleration.

Part [III](lstm_tf.ipynb) demonstrates how the model can be implemented using TensorFlow's low-level programming model in which you first define the dataflow [graph][13] and then create a TensorFlow [session][13] to run parts of the graph. In a dataflow graph, the nodes (ops) represent units of computation, and the edges (tensors) represent the data consumed or produced by a computation. Calling most functions in the TensorFlow low-level API merely adds operations and tensors to the default graph, but does not perform the actual computation. Instead, you compose these functions until you have a tensor or operation that represents the overall computation, such as performing one step of gradient descent, and then pass that object to a TensorFlow session to run the computation. This model is different from the familiar imperative model, but is a common model for parallel computing. The program is executed in [Colaboratory][12] with GPU acceleration.

It is shown that all these implementations yield results which agree with each other and with those in [Zaremba et al. (2015)][7].

---

## References
1. S. Hochreiter, and [J. Schmidhuber][1]. Long Short-Term Memory. Neural Computation, 9(8):1735-1780, 1997 

2. Christopher Olah, [Understanding LSTM networks][2], colah's blog, 27 August 2015

3. [Deep learning specialization][3], Taught by Andrew Ng, Kian Katanforoosh, and Younes Bensouda Mourri, Coursera 

4. Fei-Fei Li, Justin Johnson, and Serena Yeung, [Stanford cs231n lecture 10][4], 4 May 2017

5. Klaus Greff, Rupesh Kumar Srivastava, Jan Koutník, Bas R. Steunebrink, and Jürgen Schmidhuber, "[LSTM: A Search Space Odyssey][5]", Transactions on Neural Networks and Learning Systems, 2016 *(Errata: In version 2 of the paper on arXiv, on page 2, the first equation under "B. Backpropagation Through Time" gives the derivative of the loss with respect to yt. In that equation, there should be an over bar over z, i, f and o, denoting gradients inside the non-linear activation functions.)*

6. Andrej Karpathy, [Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy][14]
 
7. Tomáš Mikolov's web page, [Penn Tree Bank (PTB) dataset][6]

8. Wojciech Zaremba, IlyaSutskever, and Oriol Vinyals, "[Recurrent Neural Network Regularization][7]", ICLR 2015

9. TensorFlow tutorial [example][16] with eager execution
10. TensorFlow tutorial [example][15] with graph execution

11. Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar, "[On the convergence of Adam and beyond][17]", ICLR 2018 *(Errata: On 'slide 3 Algorithms', 'slide 6 Primary cause for non-convergence', and 'slide 10 AMSGrad' of [Sashank's presentation at ICLR 2018][18], in three places the exponent of beta inside the square root should be t-j instead of t-i. In one place on slide 10 in the AMSGrad update equation, the exponent of beta inside the square root should be k-j instead of k-i. Also, note that 1<=k<=t is implied.)*

[1]: http://people.idsia.ch/~juergen/
[2]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
[3]: https://www.coursera.org/specializations/deep-learning
[4]: http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf
[5]: https://arxiv.org/abs/1503.04069
[6]: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
[7]: https://arxiv.org/abs/1409.2329
[8]: https://www.python.org/
[9]: http://www.numpy.org/
[10]: https://www.tensorflow.org/
[11]: https://www.tensorflow.org/guide/eager
[12]: https://colab.research.google.com/notebooks/welcome.ipynb
[13]: https://www.tensorflow.org/guide/graphs
[14]: https://gist.github.com/karpathy/d4dee566867f8291f086
[15]: https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb
[16]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/rnn_ptb
[17]: https://openreview.net/forum?id=ryQu7f-RZ
[18]: https://www.facebook.com/iclr.cc/videos/2123421684353553/