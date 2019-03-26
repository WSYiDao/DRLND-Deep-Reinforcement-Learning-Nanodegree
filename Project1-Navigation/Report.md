# Report for Navigation
## Learning Algorithm
The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

### Q-Network
Agnet used local QNetwork and target Qnetwork with the same architectures

- input a seed to init torch.manual_seed(seed)
- nn.Linear for each layer
- state_size=37 as input nodes
- 64 of nodes in first hidden layer and  use a ReLU activation
- 64 of nodes insecond hidden layer and use a ReLU activation
- action_size=4 as output nodes

```
Network(
  (fc1): Linear(in_features=37, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=4, bias=True)
) 
```

###  Hyperparameters

- BUFFER_SIZE   - replay buffer size
- BATCH_SIZE    - minibatch size
- GAMMA         - discount factor
- TAU           - for soft update of target parameters
- LR                # learning rate 
- UPDATE_EVERY        # how often to update the network

### Plot of Rewards
chosen different hyperparameters and solve the environment with average score(each 100 episodes) 

- most quickly to receive an average reward of as least +13

```python
BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 64         
GAMMA = 0.99            
TAU = 1e-3              
LR = 5e-4               
UPDATE_EVERY = 4     

Episode 500	Average Score: 13.18  
```
- 1800 episodes to receive an  highest average reward
 
```python
BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 64         
GAMMA = 0.99            
TAU = 1e-3              
LR = 5e-5               
UPDATE_EVERY = 4        

Episode 1800	Average Score: 17.41
```

### Runing record

```python
BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 64         
GAMMA = 0.99           
TAU = 1e-3              
LR = 5e-4               
UPDATE_EVERY = 4        
agent = Agent(state_size=37, action_size=4, seed=0)
scores = dqn()

```

    Episode 100	Average Score: 1.26
    Episode 200	Average Score: 4.88
    Episode 300	Average Score: 6.96
    Episode 400	Average Score: 10.49
    Episode 500	Average Score: 13.18
    Episode 600	Average Score: 14.89
    Episode 700	Average Score: 15.18
    Episode 800	Average Score: 14.99
    Episode 900	Average Score: 15.42
    Episode 1000	Average Score: 14.89



```python
BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 64         
GAMMA = 0.99            
TAU = 1e-3              
LR = 1e-4                
UPDATE_EVERY = 4        
agent = Agent(state_size=37, action_size=4, seed=0)
scores = dqn()
```

    Episode 100	Average Score: 0.78
    Episode 200	Average Score: 4.88
    Episode 300	Average Score: 6.80
    Episode 400	Average Score: 9.25
    Episode 500	Average Score: 12.26
    Episode 600	Average Score: 14.01
    Episode 700	Average Score: 14.26
    Episode 800	Average Score: 15.72
    Episode 900	Average Score: 16.30
    Episode 1000	Average Score: 16.82



```python
BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 64         
GAMMA = 0.99            
TAU = 1e-3              
LR = 1e-4               
UPDATE_EVERY = 4        
agent = Agent(state_size=37, action_size=4, seed=0)
scores = dqn(n_episodes=1800,max_t=5000)
```

    Episode 100	Average Score: 0.67
    Episode 200	Average Score: 4.16
    Episode 300	Average Score: 7.17
    Episode 400	Average Score: 9.34
    Episode 500	Average Score: 11.71
    Episode 600	Average Score: 13.31
    Episode 700	Average Score: 14.62
    Episode 800	Average Score: 14.98
    Episode 900	Average Score: 15.49
    Episode 1000	Average Score: 16.19
    Episode 1100	Average Score: 15.73
    Episode 1200	Average Score: 16.46
    Episode 1300	Average Score: 16.02
    Episode 1400	Average Score: 16.43
    Episode 1500	Average Score: 16.58
    Episode 1600	Average Score: 16.25
    Episode 1700	Average Score: 15.33
    Episode 1800	Average Score: 15.65



```python
BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 64         
GAMMA = 0.99            
TAU = 1e-3              
LR = 5e-5               
UPDATE_EVERY = 4        
agent = Agent(state_size=37, action_size=4, seed=0)
scores = dqn(n_episodes=1800)
```

    Episode 100	Average Score: 1.28
    Episode 200	Average Score: 4.89
    Episode 300	Average Score: 7.87
    Episode 400	Average Score: 9.15
    Episode 500	Average Score: 10.81
    Episode 600	Average Score: 13.13
    Episode 700	Average Score: 14.97
    Episode 800	Average Score: 15.47
    Episode 900	Average Score: 15.25
    Episode 1000	Average Score: 16.29
    Episode 1100	Average Score: 16.09
    Episode 1200	Average Score: 16.28
    Episode 1300	Average Score: 16.19
    Episode 1400	Average Score: 15.88
    Episode 1500	Average Score: 16.38
    Episode 1600	Average Score: 16.37
    Episode 1700	Average Score: 16.88
    Episode 1800	Average Score: 17.41



```python
BUFFER_SIZE = int(1e6)  
BATCH_SIZE = 64         
GAMMA = 0.99            
TAU = 1e-3              
LR = 5e-5               
UPDATE_EVERY = 4        
agent = Agent(state_size=37, action_size=4, seed=0)
scores = dqn(n_episodes=1800)
```

    Episode 100	Average Score: 0.71
    Episode 200	Average Score: 4.02
    Episode 300	Average Score: 7.00
    Episode 400	Average Score: 8.79
    Episode 500	Average Score: 10.31
    Episode 600	Average Score: 13.31
    Episode 700	Average Score: 13.43
    Episode 800	Average Score: 14.32
    Episode 900	Average Score: 14.62
    Episode 1000	Average Score: 14.64
    Episode 1100	Average Score: 14.50
    Episode 1200	Average Score: 14.95
    Episode 1300	Average Score: 15.23
    Episode 1400	Average Score: 15.22
    Episode 1500	Average Score: 15.84
    Episode 1600	Average Score: 15.59
    Episode 1700	Average Score: 15.37
    Episode 1800	Average Score: 15.56



```python
BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 128        
GAMMA = 0.99            
TAU = 1e-3              
LR = 5e-5              
UPDATE_EVERY = 4       

agent = Agent(state_size=37, action_size=4, seed=0)
scores = dqn(n_episodes=1800)
```

    Episode 100	Average Score: 1.33
    Episode 200	Average Score: 4.93
    Episode 300	Average Score: 6.96
    Episode 400	Average Score: 10.50
    Episode 500	Average Score: 12.85
    Episode 600	Average Score: 13.35
    Episode 700	Average Score: 13.94
    Episode 800	Average Score: 15.23
    Episode 900	Average Score: 16.26
    Episode 1000	Average Score: 16.44
    Episode 1100	Average Score: 16.11
    Episode 1200	Average Score: 16.36
    Episode 1300	Average Score: 16.06
    Episode 1400	Average Score: 16.45
    Episode 1500	Average Score: 16.22
    Episode 1600	Average Score: 16.08
    Episode 1700	Average Score: 16.39
    Episode 1800	Average Score: 16.33



## Ideas for Future Work

- building networks with different architectures and find a best one for this environment
- train an agent from raw pixels instead of the 37 given state size
- try more combination of hyperparameters and find the best 
