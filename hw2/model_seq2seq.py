
# coding: utf-8

# In[1]:


#coding: utf-8
import re
import json

with open('../training_label.json') as f:
    data = json.load(f)

class WordEncoder():
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            if len(word) > 0:
                self.addWord(word)
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
    def buildIndex(self, words):
        self.word2index = {"SOS":0, "EOS":1}
        for word in words:
            self.index2word[self.n_words] = word
            self.word2index[word] = self.n_words
            self.n_words += 1
        self.index2word[self.n_words] = "OOV"
        self.word2index["OOV"] = self.n_words
        

we = WordEncoder()
for a_data in data:
    video_name = a_data['id']
    
    for label in a_data['caption']:
        s = re.sub(r"([.!?])", r"", label)
        s = re.sub(r"[^a-zA-Z]+", r" ", s)
        s = s.lower()
        we.addSentence(s)
import operator
from collections import OrderedDict
d = sorted(we.word2count.items(), key=operator.itemgetter(1), reverse=True)
for i in range(len(d)):
    if d[i][1] == 2:
        max_index = i
        break

new_d = d[:max_index]
words = []
for item in new_d:
    words.append(item[0])
we.buildIndex(words)

import numpy as np
def label2np(s, we):
    words = s.split(' ')
    words.insert(0, 'SOS')
    words.append('EOS')
    np_label = np.zeros([len(words), 1], dtype=float)
    for i, word in enumerate(words):
        if words[i] in we.word2index:
            np_label[i] = we.word2index[word]
        else:
            np_label[i] = we.word2index["OOV"]
    return np_label

# build label numpy data
## video_id as file name 
## list[label1, label2, ...]
## label(len(s), we)
import os
import torch
video_label_dict = {}
for a_data in data:
    video_name = a_data['id']

    label_list = []
    for label in a_data['caption']:
        s = re.sub(r"([.!?])", r"", label)
        s = re.sub(r"[^a-zA-Z]+", r" ", s)
        s = s.lower()

        np_label = label2np(s, we)
        label_list.append(np_label)
    video_label_dict[video_name] = label_list

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class S2VTDataset(Dataset):
    def __init__(self, video_feature_dir, transform=None, training=True):
        """
        Args:
            video_feature_dit (string): Path to the video features (80x4096).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.video_feature_dir = video_feature_dir
        self.transform = transform
        self.training = training
        self.sample = []
        video_feature_names = os.listdir(self.video_feature_dir)
        for video_feature_name in video_feature_names:
            self.add_sample(video_feature_name)
        from random import shuffle
        shuffle(self.sample)
        training_index = int(len(self.sample)*9/10)
        self.training_sample = self.sample[:training_index]
        self.validation_sample = self.sample[training_index:]
    def __len__(self):
        if self.training:
            return len(self.training_sample)
        else:
            return len(self.validation_sample)

    def __getitem__(self, idx):
        if self.training:
            return self.training_sample[idx]
        else:
            return self.validation_sample[idx]
    
    def add_sample(self, video_feature_name):
        video_feature_np = np.load(os.path.join(self.video_feature_dir, video_feature_name))
        video_name = '.'.join(video_feature_name.split('.')[:2])
        label_list = video_label_dict[video_name]
        import random
        random.seed(1)
        r_index = random.randint(0, len(label_list)-1)
        self.sample.append((video_feature_np,label_list[r_index]))
    def get_sample(self):
        return self.sample

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


# In[2]:


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        embedded = input.view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=80):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


# In[3]:
batch_size = 1
training_dataset = S2VTDataset('../training_data/feat')
training_dataloader = DataLoader(training_dataset, batch_size=batch_size,
                      shuffle=True)
validation_dataset = S2VTDataset('../training_data/feat', training=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
# In[ ]:


hidden_size = 256
encoder1 = EncoderRNN(4096, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, 2875,
                               1, dropout_p=0.1)


# In[ ]:


if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()


# In[6]:


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=80):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(80, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]
    decoder_hidden = encoder_hidden
        # Teacher forcing: Feed the target as the next input
    import random
    teacher_forcing_ratio = 1
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length-1):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                target_variable[di], decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di+1])
    else:
        # Without teacher forcing: use its own predictions as the next input
        decoder_input = target_variable[0]
        for di in range(target_length-1):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di+1])
            if ni == 1:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def evaluate(input_variable, encoder, decoder, max_length=80):
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([0]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = ['SOS']
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        
        decoded_words.append(we.index2word[ni])
        
        if ni == 1:
            break
        if len(decoded_words) > 25:
            decoded_words.append(we.index2word[1])
            break
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        
    return decoded_words
print_loss_total = 0  # Reset every print_every
learning_rate = 0.01
encoder_optimizer = optim.SGD(encoder1.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(attn_decoder1.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
print_every = 1000
best_loss = 100

model_path = '../models/10-epoch-validation-all/'
if not os.path.exists('../models'):
    os.mkdir('../models')
if not os.path.exists(model_path):
    os.mkdir(model_path)

epoches = 10
best_score=0
for epoch in range(epoches):
    bleu_score = 0
    predict_sentence = None
    label_sentence = None
    for i,sample in enumerate(training_dataloader):
        input_variable = Variable(sample[0].float().cuda().view(-1,4096))
        target_variable = Variable(sample[1].long().cuda().view(-1,1))
        
        print('training progress: {}/{}\r'.format(i, len(training_dataloader)), end='')

        loss = train(input_variable, target_variable, encoder1,
                     attn_decoder1, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        if (i+1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('\nloss(data %d): %.4f' % (i, print_loss_avg))
    for i,sample in enumerate(validation_dataloader):
        input_variable = Variable(sample[0].float().cuda().view(-1,4096))
        target_variable = Variable(sample[1].long().cuda().view(-1,1))
        
        print('validation progress: {}/{}\r'.format(i, len(validation_dataloader)), end='')
        #evaluate
        predict_words = evaluate(input_variable, encoder1, attn_decoder1)
        predict_sentence = ' '.join(predict_words[1:-1])
        from bleu_eval import BLEU
        label_words = []
        for index in target_variable:
            if index.data[0] == 1:
                break
            label_words.append(we.index2word[index.data[0]])
        label_sentence = ' '.join(label_words[1:])
        if predict_sentence == '':
            bleu_score += 0
        else:
            bleu_score += BLEU(predict_sentence, label_sentence)
    new_bleu_score = bleu_score/(len(validation_dataloader))
    torch.save(encoder1.state_dict(), model_path + 'encoder-epoch-{}'.format(epoch))
    torch.save(attn_decoder1.state_dict(), model_path + 'decoder-epoch-{}'.format(epoch))
    if new_bleu_score > best_score:
        best_score = new_bleu_score
    print('\nbleu score: {}'.format(new_bleu_score))
    print('epoch-{}: \npredict_sentence: {}\nlabel_sentence: {}'.format(epoch, predict_sentence, label_sentence))
