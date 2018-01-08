from agent_dir.agent import Agent
import scipy.misc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)

def save_data(model):
    torch.save(model.state_dict(), 'pg_model')
class PG_Model(nn.Module):
    def __init__(self):
        super(PG_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.linear1 = nn.Linear(2048, 128)
        self.linear2 = nn.Linear(128, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.linear1(x.view(1, -1)))
        x = F.softmax(self.linear2(x))
        return x

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        torch.manual_seed(5)
        self.model = PG_Model()
        self.test = args.test_pg
        if args.test_pg:
            self.model.load_state_dict(torch.load('pg_model.ckpt'))
            print('loading trained model')
        ##################
        # YOUR CODE HERE #
        ##################
        
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-4)
        self.saved_actions = []
        self.saved_log_probs = []
        self.thirty_episode_rewards = []
        self.env = env
        self.last_state = torch.zeros([1,1,80,80])
        

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
        discount = 0.99
        from itertools import count
        for i_episode in count(1):
            self.saved_actions=[]
            episode_rewards = []
            init_state = self.env.reset()
            self.last_state = torch.zeros([1,1,80,80])
            state = init_state
            for t in range(10000):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)
                episode_rewards.append(reward)
                if done:
                    break
            R = 0
            policy_loss = []
            rewards = []
            for r in episode_rewards[::-1]:
                if r != 0:
                    R = 0
                R = r + discount * R
                rewards.insert(0, R)
            rewards = torch.Tensor(rewards).cuda()
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
            for action, reward in zip(self.saved_actions, rewards):
                action.reinforce(reward)
#             for log_prob, reward in zip(self.saved_log_probs, rewards):
#                 policy_loss.append(-log_prob * reward)
            self.optimizer.zero_grad()
            autograd.backward(self.saved_actions, [None for _ in self.saved_actions])
#             policy_loss = torch.cat(policy_loss).sum()
#             policy_loss.backward(retain_graph=True)
            self.optimizer.step()
            
            
            
            if len(self.thirty_episode_rewards) < 30:
                self.thirty_episode_rewards.append(sum(episode_rewards))
                print('Episode {}\tEpisode Reward: {:.2f}'.format(
                    i_episode, sum(episode_rewards)))
            else:
                self.thirty_episode_rewards.pop(0)
                self.thirty_episode_rewards.append(sum(episode_rewards))
                print('Episode {}\tReward: {:.2f}\tAverage Reward: {:.2f}'.format(
                    i_episode, sum(episode_rewards), sum(self.thirty_episode_rewards)/30.0))
            if i_episode % 30 == 0:
                with open('rewards_2.txt', 'a+') as f:
                    f.write(str(sum(self.thirty_episode_rewards)/30.0) + '\n')
            if i_episode % 100 == 0:
                torch.save(self.model.state_dict(), 'pg_model_2_' + str(i_episode) + '.ckpt')
            if sum(self.thirty_episode_rewards)/30.0 >= 7:
                torch.save(self.model.state_dict(), 'pg_model.ckpt')
                break
            
            del self.saved_actions[:]
        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        state = np.reshape(prepro(observation), [1, 80,80])
        state = torch.from_numpy(state).float().unsqueeze(0)
        network_input = state - self.last_state
        self.last_state = state
        probs = self.model(Variable(network_input).cuda())
        action = probs.multinomial()
        if not self.test:
            self.saved_actions.append(action)
        return action.data[0][0]

