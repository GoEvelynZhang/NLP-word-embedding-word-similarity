#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
from collections import Counter


# In[2]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words = set(stop_words)


# In[3]:


df = pd.read_csv('training/training-data.1m', sep='\t', header=None)


# In[5]:


df.columns = ['text']


# In[1]:


# df.text


# In[7]:


RE_STRIP_SPECIAL_CHARS = r'[^a-zA-Z0-9\s]'
RE_WHITESPACE = r'[\s]+'
RE_NUMBER = r'[0-9]+'


# In[8]:


def extract_token(corpus):
    print('tokenizing sentences')
    result = []
    for sentence in corpus:
        s = sentence.lower()
        s = re.sub(RE_STRIP_SPECIAL_CHARS, '', s)
        # s = re.sub(RE_NUMBER, '<NUMBER>', s)
        s = re.sub(RE_NUMBER, ' ', s)
        tokenized = nltk.word_tokenize(s)
        filtered_sentence = [w for w in tokenized if not w in stop_words]
        result.append(filtered_sentence)
    return result


# In[9]:


res = extract_token(df.text)


# In[10]:


def get_vocab(tokens):
    vocab = []
    for sentence in tokens:
        for token in sentence:
            vocab.append(token)

    word_counter = Counter(vocab)
    avg_doc_length = len(vocab) / len(tokens)
    vocab = set(vocab)
    return vocab, word_counter, avg_doc_length


# In[11]:


vocab, word_counter, avg_doc_length = get_vocab(res)
vocab_size = len(vocab)
print(f'vocab size is {vocab_size}, average doc length is {avg_doc_length}')


# In[12]:


remove_words = []
remove_below = 2

for word, count in word_counter.items():
    if count < remove_below:
        remove_words.append(word)

remove_words = set(remove_words)
print(f'{len(remove_words)} words that appeared less than {remove_below} times will be substituted by <UNK>')


# In[13]:


processed_tokens = []
print('substituting words')

for sentence in res:
    temp = []
    for word in sentence:
        if word in remove_words:
            temp.append('<UNK>')
        else:
            temp.append(word)
    processed_tokens.append(temp)


# In[14]:


vocab, word_counter, avg_doc_length = get_vocab(processed_tokens)
vocab_size = len(vocab)
print(f'vocab size is {vocab_size}, new average doc length is {avg_doc_length}')


# In[15]:


# sorted(word_counter.items(), key=lambda x:x[1])


# In[16]:


word_to_idx = {w: idx for idx, w in enumerate(vocab)}
idx_to_word = {idx: w for idx, w in enumerate(vocab)}


# In[17]:


def generate_skipgram(tokens, window):
    result = []
    print(f'generating skipgrams with window size {window}')
    for idx, sentence in enumerate(tokens):
        if idx % 100000 == 0 and idx > 0:
            print(f'processed {idx} sentences')
        for idx, token in enumerate(sentence):
            for i in range(idx - window, idx, 1):
                if i >= 0:
                    skipgram = [token, sentence[i]]
                    result.append(skipgram)
            for j in range(idx + 1, idx + window + 1, 1):
                if j < len(sentence):
                    skipgram = [token, sentence[j]]
                    result.append(skipgram)
    return result


# In[18]:


def skipgram_to_idx(skipgrams, idx_dict):
    print('creating skipgram words <-> index dictionary')
    result = []
    for skipgram in skipgrams:
        result.append([idx_dict[skipgram[0]], idx_dict[skipgram[1]]])
    return result


# In[19]:


def generate_batches(skipgrams, batch_size):
    n_batches = len(skipgrams) // batch_size
    skipgrams = skipgrams[:n_batches*batch_size]
    for i in range(0, len(skipgrams), batch_size):
        context = []
        target = []
        batch = skipgrams[i:i+batch_size]
        for j in range(len(batch)):
            context.append(batch[j][0])
            target.append(batch[j][1])
        yield context, target    


# In[25]:


w = 5
skipgrams = generate_skipgram(processed_tokens, window=w)
print(f'got {len(skipgrams)} skipgrams')
skipgrams_idx = skipgram_to_idx(skipgrams, word_to_idx)


# In[26]:


import torch
from torch import nn
import torch.optim as optim


# In[27]:


class SkipgramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_dist):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.word_dist = word_dist
        
        # ("orange", "juice", observed=1), ("orange", "king", observed=0) 
        # => "orange" is context word, "juice" & "king" are target words
        self.context_embed = nn.Embedding(vocab_size, embed_dim)
        self.target_embed = nn.Embedding(vocab_size, embed_dim)
        
        self.context_embed.weight.data.uniform_(-1, 1)
        self.target_embed.weight.data.uniform_(-1, 1)
    
    def get_context_row(self, word):
        return self.context_embed(word)
    
    def get_target_row(self, word):
        return self.target_embed(word)
    
    def get_negative_samples(self, batch_size, k):
        negative_samples = torch.multinomial(self.word_dist, batch_size * k, replacement=True)
        device = "cuda" if self.target_embed.weight.is_cuda else "cpu"
        negative_samples = negative_samples.to(device)
        return self.target_embed(negative_samples).view(batch_size, k, self.embed_dim)  


# In[28]:


class SkipgramLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, context_vectors, target_vectors, negative_vectors):
        batch_size, embed_dim = context_vectors.shape
        context_vectors = context_vectors.view(batch_size, embed_dim, 1)
        target_vectors = target_vectors.view(batch_size, 1, embed_dim)
        
        observed_sample_loss = torch.bmm(target_vectors, context_vectors).sigmoid().log()
        observed_sample_loss = observed_sample_loss.squeeze()
        
        negative_sample_loss = torch.bmm(negative_vectors.neg(), context_vectors).sigmoid().log()
        negative_sample_loss = negative_sample_loss.squeeze().sum(1)
        
        return -(observed_sample_loss + negative_sample_loss).mean()


# In[29]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
word_freq = np.asarray(sorted(word_counter.values(), reverse=True))
unigram_dist = word_freq / word_freq.sum()
negative_sample_dist = torch.from_numpy(unigram_dist**(0.75) / np.sum(unigram_dist**(0.75)))

embed_dim = 100
model = SkipgramModel(vocab_size, embed_dim, negative_sample_dist).to(device)
criterion = SkipgramLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0015)

print_every = 1
epochs = 8
k = 4
batch_size = 1024

print('training started')
# train for some number of epochs
for e in range(epochs):

    counter=0
    
    # get our input, target batches
    for context_words, target_words in generate_batches(skipgrams_idx, batch_size):
        context, targets = torch.LongTensor(context_words), torch.LongTensor(target_words)
        context, targets = context.to(device), targets.to(device)

        # input, outpt, and noise vectors
        context_vectors = model.get_context_row(context)
        target_vectors = model.get_target_row(targets)
        negative_vectors = model.get_negative_samples(batch_size, k)

        # negative sampling loss
        loss = criterion(context_vectors, target_vectors, negative_vectors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        counter+=1
        if counter % 10000 == 0:
            print(counter)
        

    # loss stats
    if e % print_every == 0:
        print(f"Epoch: {e}/{epochs}")
        print("Loss: ", loss.item()) # avg batch loss at this point in training
    


# In[30]:


def output_embed(embed):
    print('writing embedding to output file')
    f = open('embedding-w5dim100-new.txt', 'w')
    for idx, word_embed in enumerate(embed):
        word = idx_to_word[idx]
        temp = word + ' '
        for value in word_embed:
            temp = temp + str(value) + ' '
        temp += '\n'
        f.write(temp)
    f.close()
    print('completed')


# In[31]:


embeddings = model.context_embed.weight.to('cpu').data.numpy()
output_embed(embeddings)


# In[ ]:




