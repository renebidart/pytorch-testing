# pytorch-testing

# Very unfinished


## Debugging CNNs is needlessly annoying
As I've started making weirder architectures I've noticed a lot more of my time is spent debugging recurring issues with gradient computations / updates. This is where I'll store all the useful pieces of code for taking a Pytorch model that is syntatically correct to one that actually trains by identifying where gradient issues in the network is.

This is based on these recommendations, with a few 
1. https://karpathy.github.io/2019/04/25/recipe/
2. https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn
3. https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
4. https://github.com/suriyadeepan/torchtest/


Includes sanity checks ensuring correct parameters are trainable, that they are updated, that these updates are correct, overfitting a batch, etc.
