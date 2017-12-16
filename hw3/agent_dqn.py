from agent_dir.agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import random
from collections import namedtuple
from itertools import count

class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = Transition
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.linear1 = nn.Linear(3136, 512)
        self.linear2 = nn.Linear(512, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.linear1(x))
        x = self.linear2(x)
        return x
class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        self.Q = DQN()
        self.target_Q = DQN()    
        if args.test_dqn:
            self.target_Q.load_state_dict(torch.load('dqn_model_MSE9800000.ckpt'))
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        torch.manual_seed(5)
        self.target_Q.load_state_dict(torch.load('dqn_model_MSE9800000.ckpt'))
        self.Q.load_state_dict(torch.load('dqn_model_MSE9800000.ckpt'))
        if torch.cuda.is_available():
            self.Q.cuda()
            self.target_Q.cuda()
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        self.optimizer = optim.RMSprop(self.Q.parameters(), lr=1e-4)
        self.memory = ReplayMemory(10000, self.Transition)
        self.env_step = 1e6
        self.env = env
        self.update = 0
        self.episode_rewards = []
        self.one_hundred_episode_rewards = []
        self.last_ep_sum = 0
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        def select_epilson_greedy_action(observation):
            sample = random.random()
            if self.env_step <= 1e6:
                epilson = 1 - (1 - 0.05)/1e6 * self.env_step
            else:
                epilson = 0.05
            if sample > epilson:
                probs = self.target_Q(Variable(torch.Tensor(observation), volatile=True).view(-1, 4, 84, 84).cuda())
                action = probs.data.max(1)[1][0]
                return torch.Tensor([action])
            else:
                action = self.env.get_random_action()
                return torch.Tensor([action])
        
        q_net_update = 0
        state = self.env.reset()
        
        LOG_EVERY_N_STEPS = 1000
        learning_start = 10000
        learning_freq = 4
        target_update_freq = 1000
        num_episodes = 1e7
        for t in count(1):
            self.env_step += 1
            if num_episodes < t:
                break
            # Select and perform an action
            recent_observations = state
            if t > learning_start:
                action = select_epilson_greedy_action(recent_observations)
            else:
                action = torch.Tensor([random.randrange(4)])
            
            obs, reward, done, _ = self.env.step(int(action[0]))
            self.episode_rewards.append(reward)
            reward = torch.Tensor([reward])
            
            
            if not done:
                next_state = obs
                self.memory.push(torch.Tensor(state), action, torch.Tensor(next_state), reward)
                state = next_state
            else:
                next_state = None
                self.memory.push(torch.Tensor(state), action, next_state, reward)
                state = self.env.reset()

                if len(self.one_hundred_episode_rewards) < 100:
                    self.one_hundred_episode_rewards.append(sum(self.episode_rewards))
                else:
                    self.one_hundred_episode_rewards.pop(0)
                    self.one_hundred_episode_rewards.append(sum(self.episode_rewards))
                self.last_ep_sum = sum(self.episode_rewards)
                self.episode_rewards = []
            
            if (t >= learning_start and t % learning_freq == 0 ):
                transitions = self.memory.sample(32)
                batch = self.Transition(*zip(*transitions))
                

                non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state))).cuda()
                tmp = []
                for s in batch.next_state:
                    tmp.append(s)
                non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]),
                                     volatile=True).view(-1, 4, 84, 84).cuda()
                state_batch = Variable(torch.cat(batch.state)).view(32, 4, 84, 84).cuda()
                action_batch = Variable(torch.cat(batch.action)).long().cuda()
                reward_batch = Variable(torch.cat(batch.reward)).cuda()
                
               
                current_Q_values = self.Q(state_batch).gather(1, action_batch.unsqueeze(1))
                next_Q_values = Variable(torch.zeros(32).type(torch.Tensor)).cuda()
                next_Q_values[non_final_mask] = self.target_Q(non_final_next_states).max(1)[0]
                next_Q_values.volatile = False
                
                expected_Q_values = (next_Q_values * 0.99) + reward_batch
                loss = F.mse_loss(current_Q_values, expected_Q_values)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.update += 1
                
                if self.update % (target_update_freq/learning_freq) == 0:
                    self.target_Q.load_state_dict(self.Q.state_dict())
            if t >= learning_start and t%100 == 0:
                with open('rewards_dqn_Mse_target.txt', 'a+') as f:
                    f.write(str(sum(self.one_hundred_episode_rewards)/100.0) + '\n')
            if t >= learning_start and t%100000 == 0:
                torch.save(self.target_Q.state_dict(), 'dqn_model_MSE_target_' + str(t) + '.ckpt')
            if t % LOG_EVERY_N_STEPS==0 and t < learning_start:
                print("Timestep %d\tnot start learing yet" % (t,))
            if t % LOG_EVERY_N_STEPS == 0 and t > learning_start:
                print('Timestep {}\tLast Episode Reward: {:.2f}\tAverage Reward: {:.2f}'.format(
                    t, self.last_ep_sum, sum(self.one_hundred_episode_rewards)/100.0))
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        probs = self.target_Q(Variable(torch.Tensor(observation), volatile=True).view(-1, 4, 84, 84).cuda())
        action = probs.data.max(1)[1][0]
        return action
