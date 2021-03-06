# Policy-gradient-based-method
Solve openai-gym environment with REINFORCE update
Environment solved if average score over 100 consecutive episodes is at least 195

# Run simulations
REINFORCE.py trains an agent with REINFORCE algorithm, then it prints scores, average score over the past 100 episodes.

# REINFORCE
Consider the "CartPole-v0" environment. We try to solve this environment with the REINFORCE algorithm (a policy gradient based method).
The algorithm uses a stochastic gradient ascent update so that convergence to a **local** optimum is assured for decreasing alpha.
REINFORCE produces slow learning as a Monte Carlo method with potential high variance.

# Neural networks as function approximator
We use a neural network to parametrize the policy. Convergence is not always achieved. Oscillations may appear. The algorithm can solve the environment in 1000-5000 episodes.

# 'Slow learning', oscillations and local optimum
When using two hidden layers (fully connected), learning generally fails (see below).

![Local optimum](local_minimum.png)

Generally speaking, as stated in Sutton (2018), backpropagation algorithm can produce good results for shallow networks, but it may not work well for deeper networks. A shallow network with one hidden layer (unit number = 4) is used to solve this environment.

![Solved](solved_shallow_ANN.png)

# Proximal policy optimization (PPO)
Below the result with the PPO algorithm with clipped surrogate objective.

![PPO](cartpole_PPO.png)
