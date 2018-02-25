
# coding: utf-8

# In[1]:


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

MODE = 'wgan-gp'  # wgan or wgan-gp
DIM = 512  # Model dimensionality
FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
# Gaussian noise, as in the plots in the paper
LAMBDA = 1  # Smaller lambda seems to help for toy tasks specifically
CRITIC_ITERS = 1  # How many critic iterations per generator iteration
BATCH_SIZE = 64  # Batch size
ITERS = 1200  # how many generator iterations to train for
use_cuda = True
torch.manual_seed(8)


# In[2]:


def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)


# In[3]:


netG = Generator()
netD = Discriminator()

netG.apply(weights_init)
netD.apply(weights_init)
if use_cuda:
    netG = netG.cuda()
    netD = netD.cuda()
optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))


fixed_noise = Variable(torch.rand(1, 100),volatile=True).cuda()


# In[4]:


from torch.utils.data import Dataset, DataLoader

class FaceImgDataset(Dataset):
    def __init__(self, img_file, tag_file, fake_tag_file):
        self.training_imgs = torch.load(img_file)
        self.training_tags = torch.load(tag_file)
        self.fake_tags = torch.load(fake_tag_file)
    def __len__(self):
        return len(self.training_imgs)
    def __getitem__(self, idx):
        return self.training_imgs[idx], self.training_tags[idx][0], self.training_tags[idx][1], self.fake_tags[idx][0], self.fake_tags[idx][1]


# In[5]:


def calc_gradient_penalty(netD, real_data, fake_data, c):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, c)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


# In[ ]:


train_hist = {}
train_hist['D_loss'] = []
train_hist['G_loss'] = []


# In[6]:


dataset = FaceImgDataset('usable_imgs.dat', 'usable_tags.dat', 'fake_tags.dat')
dataloader1 = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
BCE_loss = nn.BCEWithLogitsLoss().cuda()
CE_loss = nn.CrossEntropyLoss().cuda()
netD.train()


y_real_, y_fake_ = Variable(torch.ones(BATCH_SIZE, 1).cuda()), Variable(torch.zeros(BATCH_SIZE, 1).cuda())
for epoch in range(ITERS):
    netG.train()
    for i_batch, data in enumerate(dataloader1):
        
        imgs = Variable(data[0]).float().view(-1, 3, 64, 64).cuda()
        imgs = (imgs*2/255.0) - 1
        hair_tag = Variable(data[1].view(BATCH_SIZE, 1, -1)).float().cuda()
        eye_tag = Variable(data[2].view(BATCH_SIZE, 1, -1)).float().cuda()
        fake_hair_tag = Variable(data[3].view(BATCH_SIZE, 1, -1)).float().cuda()
        fake_eye_tag = Variable(data[4].view(BATCH_SIZE, 1, -1)).float().cuda()
        noise = Variable(torch.rand(BATCH_SIZE, 100)).cuda()
        
        #training netD
        for _ in range(CRITIC_ITERS):
            
            optimizerD.zero_grad()
            
            #train with real img real c
            D_real, C1_real, C2_real = netD(imgs)
            D_real_loss = BCE_loss(D_real, y_real_)
            
            C1_real_loss = CE_loss(C1_real, torch.topk(hair_tag, 1)[1].view(-1))
            C2_real_loss = CE_loss(C2_real, torch.topk(eye_tag, 1)[1].view(-1))
            
            G_ = netG(noise, hair_tag, eye_tag)
            D_fake, C1_fake, C2_fake = netD(G_)
            D_fake_loss = BCE_loss(D_fake, y_fake_)
            C1_fake_loss = CE_loss(C1_fake, torch.topk(hair_tag, 1)[1].view(-1))
            C2_fake_loss = CE_loss(C2_fake, torch.topk(eye_tag, 1)[1].view(-1))

            D_loss = (D_real_loss + C1_real_loss + C2_real_loss + D_fake_loss + C1_fake_loss + C2_fake_loss)
            train_hist['D_loss'].append(D_loss.data[0])

            D_loss.backward()
            optimizerD.step()
        #training netG
        
        optimizerG.zero_grad()
        G_ = netG(noise, hair_tag, eye_tag)
        D_fake, C1_fake, C2_fake = netD(G_)

        G_loss = BCE_loss(D_fake, y_real_)
        C1_fake_loss = CE_loss(C1_fake, torch.topk(hair_tag, 1)[1].view(-1))
        C2_fake_loss = CE_loss(C2_fake, torch.topk(eye_tag, 1)[1].view(-1))
        G_loss = G_loss + (C1_fake_loss + C2_fake_loss)
        train_hist['G_loss'].append(G_loss.data[0])

        G_loss.backward()
        optimizerG.step()

        print('Epoch {}/{}, Batch {}/{} \r'.format(epoch, ITERS, i_batch, len(dataloader1)-1), end='')
    print('')    
    print("Epoch : {}".format(epoch))
    print("D_loss : {}".format(sum(train_hist['D_loss'][-215:])/len(dataloader1)-1))
    print("G_loss : {}".format(sum(train_hist['G_loss'][-215:])/len(dataloader1)-1))
    print('\n')
    

    
        
    if epoch % 10 == 9:
        torch.save(netG.state_dict(), 'models/deepD3/G_model_epoch{}'.format(epoch))
        torch.save(netD.state_dict(), 'models/deepD3/D_model_epoch{}'.format(epoch))
        torch.save(train_hist, 'training_hist_deepD3.log')
    netG.eval()
    for i in range(5):
        sample_hair_tag = torch.zeros([1, 1, 12]).cuda()
        sample_hair_tag[0][0][i] = 1
        sample_eye_tag = torch.zeros([1, 1, 11]).cuda()
        sample_eye_tag[0][0][i] = 1
        img_sample = netG(fixed_noise, Variable(sample_hair_tag), Variable(sample_eye_tag))
        img_sample = torch.round((img_sample+1)*255/2.0)
        img_sample = img_sample.cpu().data.numpy()
        img_sample = img_sample.reshape([64,64,3])
        scipy.misc.imsave('imgs/deepD3/sample_img{}_{}.jpg'.format(i,epoch), img_sample)

            
            

