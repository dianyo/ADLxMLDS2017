
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Generator, Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[2]:


use_cuda=True
netG = Generator()

if use_cuda:
    netG = netG.cuda()


# In[3]:


hair_types = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
    'green hair', 'red hair', 'purple hair', 'pink hair',
    'blue hair', 'black hair', 'brown hair', 'blonde hair']
eyes_types = ['gray eyes', 'black eyes', 'orange eyes',
    'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
    'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

test_hair_tags = ['blue hair', 'black hair', 'red hair']
test_eyes_tags = ['red eyes', 'blue eyes', 'green eyes']

def tag2vector(tag, feature_type):
    if feature_type == 'hair':
        v = np.zeros(len(hair_types))
        v[hair_types.index(tag)] = 1
    else:
        v = np.zeros(len(eyes_types))
        v[eyes_types.index(tag)] = 1
    return v


# In[20]:


netG.load_state_dict(torch.load('G_model_epoch469'))
torch.manual_seed(2)
fixed_noise = Variable(torch.rand(5,1,100),volatile=True).cuda()
for i in range(3):
    sample_hair_tag = torch.from_numpy(tag2vector(test_hair_tags[i], 'hair')).float().cuda().view(1,1,12)
    sample_eye_tag = torch.from_numpy(tag2vector(test_eyes_tags[i], 'eyes')).float().cuda().view(1,1,11)
    for j in range(5):
        img_sample = netG(fixed_noise[j], Variable(sample_hair_tag), Variable(sample_eye_tag))
        img_sample = torch.round((img_sample+1)*255/2.0)
        img_sample = img_sample.cpu().data.numpy()
        img_sample = img_sample.reshape([64,64,3])
        scipy.misc.imsave('samples/sample_{}_{}.jpg'.format(i+1, j+1), img_sample)

