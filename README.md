# Wormhole Dynamics in Deep Neural Networks



This work investigates the generalization behavior of deep neural networks (DNNs), focusing on the phenomenon of "fooling examples," where DNNs confidently classify inputs that appear random or meaningless to humans. To explore this phenomenon, we introduce an analytical framework based on maximum likelihood estimation, bypassing the need for gradient-based optimization or explicit labels. Our analysis reveals that DNNs operating in an overparameterized regime exhibit a collapse in the output feature space. While this collapse improves network generalization, the addition of more layers eventually leads to a singularity, where the model learns trivial solutions by mapping distinct inputs to the same output, yielding zero loss. Further exploration demonstrates that this singularity can be resolved through our newly derived "wormhole" solution. The wormhole solution, when applied to arbitrary fooling examples, enables the reconciliation of meaningful labels with random ones and provides a novel perspective on shortcut learning. These findings offer deeper insights into DNN generalization, particularly in unsupervised settings, and suggest promising directions for enhancing model robustness in practical applications.





# this work currently under review
