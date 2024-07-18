# Why Do Deep Neural Networks Converge? Trajectory-Based Insights and Analysis



In this work, we formalize neural networks within
an extended vector space that incorporates the layer component.
This formalization enables trajectory-based analysis of the input
vector as it evolves through each layer of a neural network.
We demonstrate how the convergence of a neural network
can be accelerated in an overparameterized regime, ensuring
convergence to a global minimum. These findings, applicable
to both linear and non-linear neural networks (with non-linear
activation functions), provide new insights into simplicity bias
and shortcut learning. Specifically, we show that overparameterized
neural networks can achieve rapid convergence and
perfectly fit random noise, even without explicit regularization
and gradient computation. This suggests that the convergence
of neural networks may not solely rely on the gradient-based
optimization methods commonly employed in contemporary deep
learning models. While the ability of deep neural networks to
fit random noise is well-known, our discovery of convergence
behavior—characterized by the symmetric maximization and
minimization of pair-wise distances between samples used for
optimization and testing—has not been previously studied and
analysed.
