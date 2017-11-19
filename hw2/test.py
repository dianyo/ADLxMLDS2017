import numpy as np
import json

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
import sys
import os
data_dir = sys.argv[1]
test_output = sys.argv[2]
peer_output = sys.argv[3]

with open(os.path.join(data_dir,'testing_label.json')) as f:
    data = json.load(f)




import os
import torch
import pickle 
from utils import WordEncoder
we = torch.load('wordencoder')

import re
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


from torch.utils.data import Dataset, DataLoader
import torch
class S2VTTestingDataset(Dataset):
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
    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]
    
    def add_sample(self, video_feature_name):
        video_feature_np = np.load(os.path.join(self.video_feature_dir, video_feature_name))
        video_name = '.'.join(video_feature_name.split('.')[:2])
        self.sample.append((video_feature_np,video_name))
    def get_sample(self):
        return self.sample
# encoder, decoder

from utils import EncoderRNN, AttnDecoderRNN
# evaluate
batch_size = 1
testing_dataset = S2VTTestingDataset(data_dir + 'testing_data/feat')
testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size,
                      shuffle=False)
peer_dataset = S2VTTestingDataset(data_dir + 'peer_review/feat')
peer_dataloader = DataLoader(peer_dataset, batch_size=batch_size,
                            shuffle=False)
# In[ ]
encoder_path = 'encoder-epoch-1'
decoder_path = 'decoder-epoch-1'

hidden_size = 256
encoder1 = EncoderRNN(4096, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, 2875,
                               1, dropout_p=0.1)
encoder1.load_state_dict(torch.load(encoder_path))
attn_decoder1.load_state_dict(torch.load(decoder_path))


# In[ ]:

use_cuda = torch.cuda.is_available()
if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

encoder1.eval()
attn_decoder1.eval()

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
from torch.autograd import Variable

predict_tuple = []
for i,sample in enumerate(testing_dataloader):

    input_variable = Variable(sample[0].float().cuda().view(-1,4096))
    name = sample[1][0]
    print('testing progress: {}/{}\r'.format(i, len(testing_dataloader)), end='')
    #evaluate
    predict_words = evaluate(input_variable, encoder1, attn_decoder1)
    predict_sentence = ' '.join(predict_words[1:-1])

    predict_tuple.append((name, predict_sentence))

with open(test_output, 'w+') as f:
    for name, sentence in predict_tuple:
        f.write('{},{}\n'.format(name, sentence))

predict_tuple = []
for i,sample in enumerate(peer_dataloader):

    input_variable = Variable(sample[0].float().cuda().view(-1,4096))
    name = sample[1][0]
    print('testing progress: {}/{}\r'.format(i, len(peer_dataloader)), end='')
    #evaluate
    predict_words = evaluate(input_variable, encoder1, attn_decoder1)
    predict_sentence = ' '.join(predict_words[1:-1])

    predict_tuple.append((name, predict_sentence))

with open(peer_output, 'w+') as f:
    for name, sentence in predict_tuple:
        f.write('{},{}\n'.format(name, sentence))
