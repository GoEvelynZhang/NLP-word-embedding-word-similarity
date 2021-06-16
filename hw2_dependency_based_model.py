#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import re
import csv
from collections import Counter
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words = set(stop_words)


# In[5]:


def load_sentence(file):
    data1 = open(file, "r")
    file1 = data1.read()
    data1.close()
    result_sentences = file1.split("\n")
    df = pd.DataFrame(result_sentences, columns = ["text"])
    return df
  


# In[6]:


df = load_sentence("data/training/training-data.1m")
df.head()


# In[9]:


RE_STRIP_SPECIAL_CHARS = r'[^a-zA-Z0-9\s]'
RE_WHITESPACE = r'[\s]+'
RE_NUMBER = r'[0-9]+'
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


# In[10]:


res = extract_token(df.text) #collection of tokenized sentence(with preprocess on text)


# ### build V (substituting the lower fequency words), and build the V reference table
# * V: set of non-duplicated words we use in building (w,c) pairs
# * word_counting: dictiorcy for each word and its frquency in the training set, used in negative sampling and remove low-frequency words to deal with unknown

# In[ ]:


# def get_vocab(tokens):
#     vocab = []
#     for sentence in tokens:
#         for token in sentence:
#             vocab.append(token)

#     word_counter = Counter(vocab)
#     avg_doc_length = len(vocab) / len(tokens)
#     vocab = set(vocab)
#     return vocab, word_counter, avg_doc_length


# In[ ]:


# def build_vocub(remove_threshold, res):
#     vocab, word_counter, avg_doc_length = get_vocab(res)
#     remove_words = []
#     remove_below = remove_threshold

#     for word, count in word_counter.items():
#         if count < remove_below:
#             remove_words.append(word)

#     remove_words = set(remove_words)
#     processed_tokens = []
#     print('substituting words...')

#     for sentence in res:
#         temp = []
#         for word in sentence:
#             if word in remove_words:
#                 temp.append('<UNK>')
#             else:
#                 temp.append(word)
#         processed_tokens.append(temp)
#     vocab, word_counter, avg_doc_length = get_vocab(processed_tokens)
#     vocab_size = len(vocab)
#     print(f'vocab size is {vocab_size}, new average doc length after substituting low freuency token is {avg_doc_length}')
#     return vocab, word_counter, avg_doc_length, remove_words


# In[ ]:


# vocab, word_counter, avg_doc_length,remove_words = build_vocub(4, res)


# In[ ]:


# word_to_idx = {w: idx for idx, w in enumerate(vocab)}
# idx_to_word = {idx: w for idx, w in enumerate(vocab)}


# ### generate D for sytactic context

# In[11]:


def load_syntex_annotation(file_name):
    data = open(file_name, "r")
    file = data.read()
    data.close()
  # result = np.zeros((len(file),10))
    result_sentences = file.split("\n\n")
    result_tokenize = []
    for i in result_sentences:
        cur_sentence = []
        cur = i.split("\n")
        for j in cur:
            cur_sentence.append(j.split("\t"))
        result_tokenize.append(cur_sentence)
    return result_tokenize[:-1]


# In[12]:


result_tokenize = load_syntex_annotation("data/training/training-data.1m.conll")


# ### generate the syntax_idx for each sentence in training dataset
# 
# 
# 

# In[ ]:


def reconstruct(idx):
    return " ".join([i[1] for i in result_tokenize[idx]])


# In[ ]:


import csv
def creat_and_save_syntax_idx(df,result_tokenize):
    rebuild_col = [reconstruct(i) for i in range(len(result_tokenize))]
    counter  = 0
    text_to_syntex_idx = {}
    for i in range(len(df.text)):
        try:
            idx = rebuild_col.index(df.text[i])
        # print(idx)
            text_to_syntex_idx[i] = idx
        except:
            continue
        if i %9999 == 0:
            with open('syntax_index_%s.csv'%(i//9999), 'w') as csv_file:  
                writer = csv.writer(csv_file)
                for key, value in text_to_syntex_idx.items():
                    writer.writerow([key, value])
            text_to_syntex_idx = {}
    # file = pd.DataFrame(text_to_syntex_idx )
    text_to_syntex_idx = {}
    with open('index syntax file/syntax_index_%s.csv'%(i), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in text_to_syntex_idx.items():
            writer.writerow([key, value])


# In[ ]:


# creat_and_save_syntax_idx(df,result_tokenize)


# In[ ]:


# file = open('index syntax file/syntax_index_%s.csv')


# In[13]:


#load the syntax_idx
def load_syntex_idx_dict():
    text_to_syntex_idx = {}
    for i in range(100):
        try:
            with open('index syntax file/syntax_index_%s.csv'%(i)) as csv_file:
                reader = csv.reader(csv_file)
                mydict = dict(reader)
                text_to_syntex_idx.update(mydict)
        except:
            continue
    print("%s of the 1 million training text instance is annotated"%len(set(text_to_syntex_idx.keys())))
    return text_to_syntex_idx


# In[14]:


text_to_syntex_idx = load_syntex_idx_dict()


# In[15]:


def generate_dependency_based(tokens,result_tokenize):
    result = []
    count = 0
    featurized_token=[]
    for idx, sentence in enumerate(tokens):
        if idx % 100000 == 0:
            print("completed %s dependency-based feature extraction"%(idx))
        try:
            cur_annotated_lst = result_tokenize[int(text_to_syntex_idx[str(idx)])]
            ref_col = [int(i[6]) for i in cur_annotated_lst]
            word_lst = [i[1] for i in cur_annotated_lst]
            word_process_lst = []
            for s in word_lst:
                s1 = s.lower()
                s1 = re.sub(RE_STRIP_SPECIAL_CHARS, '', s1)
                s1 = re.sub(RE_NUMBER, ' ', s1)
                if s1 not in stop_words:
                    word_process_lst.append(s1.strip())
                else:
                    word_process_lst.append(' ')
            temp_idx = []
            cur_pair_col =[]
            for idx_token, token in enumerate(sentence):
                cur_word_idx = word_process_lst.index(token)
                refer_idx = int(cur_annotated_lst[cur_word_idx][6])
                if (word_process_lst[refer_idx -1]!= ' ') and (word_process_lst[refer_idx -1]!= ''):
                    cur_pair = [token,word_process_lst[refer_idx -1]]
                    cur_pair_col.append(cur_pair)
                    temp_idx.append(cur_word_idx)
                    temp_idx.append(refer_idx -1)
            
                chosen_idx_col = np.where(np.array(ref_col) == cur_word_idx + 1)[0]
                for j in chosen_idx_col:
                    if (word_process_lst[j] != ' ') and (word_process_lst[j] !=''):
                        cur_pair = [token,word_process_lst[j]]
                        cur_pair_col.append(cur_pair)
                        temp_idx.append(j)
                        temp_idx.append(cur_word_idx)
            vocub_word = [word_process_lst[i] for i in temp_idx]
            result.extend(cur_pair_col)
#             print(vocub_word)
            featurized_token.extend(vocub_word)
            count += 1
           
        except:
            continue
    flatten_result = set([item for sublist in result for item in sublist])
    print(set(featurized_token) -flatten_result )
    print(flatten_result - set(featurized_token))
    #replace element of pairs in result belong to remove_words with'<UNK>' => generalize words on 'Unknown words' with all the context of low-frequency words
    print("%s sentences participated in the dependecy-based feature extraction"%(count))
    
    
    
    return result, featurized_token


# In[16]:


def get_vocab(tokens):
#     vocab = []
#     for sentence in tokens:
#         for token in sentence:
#             vocab.append(token)

    word_counter = Counter(tokens)
#     avg_doc_length = len(vocab) / len(tokens)
    vocab = set(tokens)
    return vocab, word_counter


# In[17]:


def build_vocub(remove_threshold, featurized_token):
    vocab, word_counter= get_vocab(featurized_token)
    remove_words = []
    remove_below = remove_threshold
    for word, count in word_counter.items():
        if count < remove_below:
            remove_words.append(word)

    remove_words = set(remove_words)
    print('substituting %s words'%(len(remove_words)))
    new_featurized_token = []
    for token in featurized_token:
        if token in remove_words:
            new_featurized_token.append('<UNK>')
        else:
            new_featurized_token.append(token)
    new_vocab, new_word_counter = get_vocab(new_featurized_token)
    vocab_size = len(set(new_vocab))
    print(f'vocab size is {vocab_size}')
    return new_vocab, new_word_counter, remove_words,new_featurized_token


# In[18]:


def feature_adjust_unknown(featurized_token, remove_words):
    result = []
    for sentence in featurized_token:
        temp =[]
        for word in sentence:
            if word in remove_words:
                temp.append('<UNK>')
            else:
                temp.append(word)
        result.append(temp)
    return result
            


# In[19]:



    
result, featurized_token = generate_dependency_based(res,result_tokenize)


# In[20]:


vocab, word_counter, remove_words,new_featurized_token = build_vocub(2, featurized_token)


# In[21]:


result_adjusted = feature_adjust_unknown(result, remove_words)


# In[29]:



print("The adjusted training dependecy-based (c,w) pair has size of",len(result_adjusted))


# In[30]:


flatten_result = set([item for sublist in result_adjusted for item in sublist])
print(set(new_featurized_token ) -flatten_result )
print( flatten_result- set(new_featurized_token ) )


# In[31]:


vocab_size = len(vocab)


# In[32]:


word_to_idx = {w: idx for idx, w in enumerate(vocab)}
idx_to_word = {idx: w for idx, w in enumerate(vocab)}


# * first build a (c,w) pair collection, since not each w exist a pair that's featurized, return the featurized_token
# * using the featurized_token to build the vocab, calculate the frequency of each token and replace the token(smaller than threshold) to be "unknown"\
# * recalculate the vocab and word frequency with substitution(with unknown), and build the word to index dict
# * adjsut the (c,w) pairs with substitution

# In[ ]:


# result = generate_dependency_based(res,remove_words)


# ### build (c,w) pair in idx format pairs

# In[33]:


def skipgram_to_idx(dependency_based, idx_dict):
    print('creating dependency_based (w,c) pairs <-> index dictionary')
    result = []
    
    for db in dependency_based:
        result.append([idx_dict[db[0]], idx_dict[db[1]]])

    return result


# In[34]:


# 'nonidentifying' in remove_words


# In[35]:


db_idx= skipgram_to_idx(result_adjusted, word_to_idx)



# In[50]:


#temp save
file_out = pd.DataFrame(result_adjusted)
file_out.to_csv("temp result/dependecy_based_pairs.csv", index = False)


# In[51]:


# 
file_out2 = pd.DataFrame(db_idx)
file_out2.to_csv("temp result/dependecy_based_db_idx_pairs.csv", index = False)


# ## Build the training model

# In[36]:


import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
from torch.utils.data import TensorDataset, DataLoader


# In[37]:


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


# In[38]:


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


# In[39]:


training_idx= np.array(db_idx) 
print(training_idx.shape)


# In[40]:


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


# In[41]:


# training_ratio = 0.0001

# selected_idx =  torch.randperm(training_idx.size(0))[:int(training_idx.size(0)*training_ratio)]
training_size = len(db_idx)
selected_dataset = db_idx[:training_size]

print("selected training size %s"%(len(selected_dataset)))


# In[ ]:





# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
word_freq = np.asarray(sorted(word_counter.values(), reverse=True))
unigram_dist = word_freq / word_freq.sum()
negative_sample_dist = torch.from_numpy(unigram_dist**(0.75) / np.sum(unigram_dist**(0.75)))
batch_size = 16384
embed_dim = 300
model = SkipgramModel(vocab_size, embed_dim, negative_sample_dist).to(device)
criterion = SkipgramLoss()
optimizer = optim.Adam(model.parameters())
#, lr=0.0015
print_every = 1
epochs = 50
k = 4


print('training started')
# train for some number of epochs
for e in range(epochs):

    counter=0
    
    # get our input, target batches
    for context_words, target_words in generate_batches(selected_dataset, batch_size):
        context, targets = torch.LongTensor(context_words), torch.LongTensor(target_words)
        context, targets = context.to(device), targets.to(device)

        # input, outpt, and noise vectors
        context_vectors = model.get_context_row(context)
        target_vectors = model.get_target_row(targets)
        negative_vectors = model.get_negative_samples(batch_size, k)

        # negative sampling loss
        loss = criterion(context_vectors, target_vectors, negative_vectors)
#         print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        counter+=1
        if counter % 10000 == 0:
            print(counter)
        

    # loss stats
    if e % print_every == 0:
        print(f"Epoch: {e+1}/{epochs}")
        print("Loss: ", loss.item()) # avg batch loss at this point in training
    


# ### output the embedding and check from samples

# In[43]:


def output_embed(embed):
#     print('writing embedding to output file')
    f = open('embedding_5.txt', 'w')
    for idx, word_embed in enumerate(embed):
        word = idx_to_word[idx]
        temp = "".join(word.split()) + ' '
        for value in word_embed:
            temp = temp + str(value) + ' '
        temp += '\n'
        f.write(temp)
    f.close()
    print('completed')


# In[44]:


embeddings = model.context_embed.weight.to('cpu').data.numpy()
output_embed(embeddings)


# ### get the development word pairs similarity
# 
# 
# 

# In[45]:


import numpy as np
import pandas as pd
def preprocess_word(s):
    s1 = s.lower()
    s1 = re.sub(RE_STRIP_SPECIAL_CHARS, '', s1)
    s1 = re.sub(RE_NUMBER, ' ', s1)
    s1 = s1.strip()
    if s1 in stop_words:
        s1 = ''
    else: 
        s1 = s1.strip()
    return s1
def get_dev_similarity(embed_path, output_path):
    def read_embedding(path):
        embedding = {}
        dim = None
        for row in open(path):
            word, *vector = row.split()
#             try:
            embedding[word] = [float(x) for x in vector]
#             except:
#                 continue

            if dim and len(vector) != dim:

                print("Inconsistent embedding dimensions!", file = sys.stderr)
                sys.exit(1)

            dim = len(vector)

        return embedding, dim



    E, dim = read_embedding(embed_path)
    pairs = pd.read_csv("data/similarity/dev_x.csv", index_col = "id")
    cal_similarity = []
    for w1, w2 in zip(pairs.word1, pairs.word2):
        w1 = preprocess_word(w1)
        w2 = preprocess_word(w2)
        if w1 not in E.keys():
            cur1 = '<UNK>'
        else:
            cur1 = w1
        if w2 not in E.keys():
            cur2 = '<UNK>'
        else:
            cur2 = w2
        cal_similarity.append(np.dot(E[cur1], E[cur2]))
    pairs["similarity"] = cal_similarity
    # [np.dot(E[w1], E[w2])
    #     for w1, w2 in zip(pairs.word1, pairs.word2)]
    print(len(cal_similarity))
    del pairs["word1"], pairs["word2"]

    # print("Detected a", dim, "dimension embedding.", file = sys.stderr)
    pairs.to_csv(output_path)


# In[46]:


get_dev_similarity("embedding_5.txt", "prediction_5.csv")


# ### evaluate the development set correlation
# 
# 
# 

# In[47]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr
def evaluate_correlation(prediction_path):
    pred = pd.read_csv(prediction_path, index_col = "id")
    dev = pd.read_csv("data/similarity/dev_y.csv", index_col = "id")
    pred.columns = ["predicted"]
    dev.columns = ["actual"]
    data = dev.join(pred)
    print("Correlation:", spearmanr(data).correlation)


# In[48]:


evaluate_correlation("prediction_5.csv")


# In[49]:


words = {'acronymic', 'implicational', 'shouter', 'fractures', 'endurable',
'season', 'interplanetary', 'panic', 'fastness', 'disinvestment', 'up-to-date',
'admiralty', 'murder', 'loss', 'rejoinders', 'cash', 'metal', 'exhibit',
'exterminate', 'disinheritance', 'churchs', 'discriminate', 'vulgarism',
'recourse', 'deciphering', 'partible', 'marriage', 'meet', 'houseful',
'unemotional', 'nest', 'sodium', 'carnivore', 'circumscribes', 'listeners',
'continuance', 'cylindrical', 'undeniable', 'preschoolers', 'courteous',
'establishment', 'submerging', 'grow', 'improvised', 'shrink', 'sandbag',
'prudent', 'sexually', 'france', 'moralist', 'affiliation', 'householders',
'evidently', 'provisionally', 'sun', 'monotony', 'trusteeship', 'conversely',
'indirectness', 'transverse', 'seafood', 'incensing', 'immigrating', 'rainless',
'harpsichord', 'rational', 'intentionality', 'hypocrisy', 'common',
'distribute', 'undefined', 'combust', 'religionist', 'occlusion', 'recognition',
'medal', 'partiality', 'wrong', 'paradoxical', 'benchmark', 'episode',
'unbroken', 'brightness', 'unconsolidated', 'payment', 'unmentionables', 'skip',
'school', 'motherless', 'soldier', 'tell', 'seal', 'hole', 'sheikhdoms',
'basic', 'peripheral', 'shower', 'omnipotence', 'retrying', 'boisterously',
'detestable', 'defecation', 'child', 'abandon', 'inversions', 'leagued',
'pretenders', 'soap', 'chance', 'mile', 'deny', 'capitalised', 'terror',
'inventively', 'finance', 'evacuated', 'prostatic', 'bend', 'worthless',
'nation', 'variable', 'refer', 'rank', 'doctor', 'death', 'fret', 'policy',
'back', 'furnace', 'importances', 'affectional', 'canyon', 'bronchus',
'possible', 'civilise', 'interpenetrate', 'villainous', 'production', 'market',
'comportment', 'conjecture', 'algebraist', 'contortionists', 'translocating',
'entrapping', 'defrauding', 'brave', 'artist', 'smile', 'shoulders',
'attributions', 'legalism', 'pervert', 'regiment', 'possibility', 'bachelors',
'citizen', 'decapitated', 'conflict', 'cold', 'return', 'naughtiness',
'careful', 'month', 'hypertext', 'cock', 'hyperextension', 'transforming',
'glittery', 'text', 'procreation', 'pessimist', 'typify', 'alcohol',
'disturbing', 'marginality', 'noticeable', 'preventive', 'charm', 'libelous',
'homosexual', 'bread', 'covariant', 'measurements', 'congeniality',
'self-fulfillment', 'take', 'initiation', 'ranch', 'deposes', 'harmony',
'drafting', 'butter', 'feel', 'instrumentality', 'ingroup', 'heterosexual',
'curious', 'recess', 'anxiety', 'make', 'assembly', 'distrustful', 'asteroidal',
'leading', 'copy', 'corroding', 'solemnity', 'pinpointed', 'manacles',
'brigadier', 'tasteful', 'utility', 'responsible', 'containerful',
'conscientiousness', 'circumcisions', 'unconcern', 'insurgent', 'credit',
'displeased', 'get', 'violating', 'shepherd', 'refered', 'unprecedented',
'gender', 'certificate', 'publication', 'alleviated', 'unsighted', 'dysentery',
'needlepoint', 'abductor', 'deceitful', 'autopilot', 'enunciating', 'excessive',
'boastful', 'coffee', 'culture', 'preparation', 'sell', 'student', 'inexpert',
'subserve', 'desiccating', 'irredeemable', 'constitute', 'evaluate',
'unostentatious', 'performance', 'propagate', 'mushroomed', 'trouble',
'restraint', 'grocery', 'circumvent', 'match', 'book', 'unannounced',
'dreadnought', 'appearance', 'laugh', 'disagree', 'measure', 'subletting',
'body', 'unstuff', 'peace', 'excitations', 'concentration', 'international',
'Japanese', 'advised', 'astronomer', 'sound', 'deforming', 'material',
'fetishism', 'syntaxes', 'inorganic', 'counterfeit', 'devilishly', 'carry',
'mental', 'spot', 'restrainer', 'live', 'socialise', 'reassess', 'scholarship',
'foreclosed', 'prevent', 'missile', 'abuse', 'edible', 'germinate', 'imitate',
'criticality', 'initialise', 'cabbage', 'suspiciousness', 'fruitful', 'respite',
'sign', 'sweet', 'preconceptions', 'badness', 'tax', 'memorialize', 'tableware',
'tube', 'traitorous', 'hire', 'mouth', 'admitting', 'append', 'differentia',
'monotype', 'remind', 'immobile', 'interrelated', 'incommensurable', 'heavy',
'socialites', 'intermingles', 'Wednesday', 'voraciously', 'muster',
'insufficiency', 'england', 'minority', 'language', 'attack', 'undeviating',
'covering', 'replicate', 'unimpressed', 'impressionable', 'defiles', 'reform',
'oxide', 'skillfulness', 'bestowal', 'wheaten', 'real', 'castled', 'falsifier',
'attributable', 'mercantile', 'implausible', 'journey', 'site', 'gem',
'disturbances', 'enfolded', 'astonish', 'algebraic', 'available', 'rounded',
'repulses', 'vulnerable', 'unmelted', 'walloper', 'pedaler', 'cut',
'wilderness', 'sport', 'squishing', 'nakedness', 'district', 'inheritor',
'intragroup', 'transmigrating', 'argue', 'involvement', 'autoregulation',
'health', 'apparition', 'eventful', 'fauna', 'choose', 'area', 'trial',
'ethnic', 'ebb', 'similar', 'localise', 'analyzed', 'translocate',
'conspicuousness', 'triclinic', 'multidimensional', 'guillotine', 'network',
'exaction', 'entrench', 'scandinavian', 'building', 'obtainment', 'diagonal',
'sympathized', 'asian', 'bendability', 'substance', 'sorcery', 'kingship',
'absconding', 'kind', 'galvanize', 'artifact', 'spill', 'avenue', 'process',
'life', 'diffidence', 'hill', 'estrogenic', 'reproduction', 'vector', 'compel',
'changeableness', 'history', 'casteless', 'plundered', 'seafaring', 'brightly',
'page', 'situation', 'defiant', 'subspecies', 'chauvinist', 'room', 'chromatic',
'stupid', 'bank', 'lay', 'unwrap', 'kid', 'poisoning', 'animality',
'sublieutenant', 'transvestitism', 'rebellious', 'overshoe', 'crudeness',
'penis', 'tunneled', 'assessments', 'independences', 'short-change',
'conformations', 'distinction', 'censoring', 'dirty', 'fluidity', 'automatic',
'critic', 'untilled', 'discrimination', 'brisker', 'nonperformance',
'friendliness', 'unfavorable', 'habitable', 'bleach', 'paparazzo', 'scrutiny',
'crane', 'destroyers', 'pronounce', 'transformation', 'self-improvement',
'humanness', 'unisons', 'concordance', 'bengali', 'heat', 'resource',
'shepherded', 'thrombosis', 'trioxide', 'boldness', 'precedent', 'backwardness',
'cofactor', 'scandalize', 'enunciates', 'liveable', 'comfortable', 'deep',
'commissions', 'acceptance', 'embezzle', 'upset', 'replacements', 'evaporate',
'inessential', 'protestant', 'intercede', 'recommendation', 'organismal',
'animalism', 'principality', 'raw', 'combusting', 'transalpine', 'positioners',
'hike', 'reprints', 'nonconscious', 'obstructive', 'disadvantaged',
'infectiously', 'regenerate', 'note', 'interlink', 'naturalise', 'stewardship',
'gracefulness', 'decision', 'synchronic', 'consumptive', 'nontoxic', 'expose',
'greenness', 'procreated', 'despoil', 'thinness', 'entrust', 'repeat', 'nurse',
'terrorize', 'join', 'internationalize', 'dissociations', 'consubstantial',
'microcircuit', 'empower', 'disability', 'sexy', 'stockers', 'prize',
'prejudging', 'seepage', 'photographer', 'intertwining', 'delimited',
'transmigrated', 'combatted', 'religious', 'poison', 'shape', 'behaviorist',
'suspect', 'microfilm', 'suppressor', 'deflate', 'exchangeable', 'stitch',
'seller', 'five', 'afghani', 'scampering', 'flee', 'mammal', 'line', 'emerald',
'uncomfortable', 'enslaves', 'recitalist', 'governance', 'moderate',
'attendances', 'innovativeness', 'unparented', 'American', 'commandership',
'purveying', 'scientist', 'weaken', 'steepen', 'informal', 'technology',
'spread', 'deletion', 'universe', 'submarine', 'disquieting', 'disgust',
'breathe', 'uncomprehending', 'germanic', 'retraction', 'noon', 'translunar',
'orchestrations', 'unpersuasive', 'genuinely', 'detectable', 'lengthy',
'energetic', 'invigorating', 'negotiable', 'plate', 'survivalist', 'entombment',
'fundamentalism', 'bounce', 'microfossils', 'Harvard', 'develop', 'circumpolar',
'example', 'concurrencies', 'modesty', 'statement', 'consigning', 'cheap',
'hive', 'gene', 'weirdly', 'blitzed', 'performances', 'thousand', 'squirt',
'singe', 'desirable', 'unnecessary', 'effort', 'race', 'serve', 'venomous',
'highjacking', 'wrestle', 'serial', 'picture', 'unfeathered', 'lastingly',
'rack', 'rook', 'help', 'trace', 'metabolism', 'sociable', 'luxuriance',
'spacewalker', 'wheel', 'defeating', 'premisses', 'obviousness', 'condensing',
'association', 'fortitude', 'incommutable', 'system', 'lie', 'susceptible',
'force', 'standardize', 'brand-newness', 'enthuse', 'algebra', 'arrange',
'laboratory', 'branch', 'intense', 'bewitchment', 'check', 'computation',
'anamorphosis', 'subdividing', 'hybridise', 'shrieks', 'traditionalism',
'continue', 'repurchases', 'disclosure', 'program', 'unrepeatable', 'word',
'stroked', 'survey', 'beginning', 'cry', 'rotational', 'engrave', 'exhibited',
'confess', 'residing', 'warship', 'rudderless', 'insurrectional',
'unaccessible', 'content', 'cockerel', 'attested', 'woodland', 'calendar',
'secularist', 'painkillers', 'ganging', 'equivalence', 'interlayers',
'undetectable', 'assay', 'biographer', 'prophetic', 'standard', 'brandy',
'inabilities', 'lover', 'protectorship', 'employed', 'nonviable', 'distressful',
'prisoner', 'unformed', 'unicycling', 'contraception', 'transducers',
'codefendants', 'perform', 'sugar', 'thermonuclear', 'ill', 'microvolts',
'dissolved', 'murderer', 'lilt', 'colorful', 'official', 'painter', 'position',
'summonings', 'disjunct', 'outfoxed', 'flattery', 'fill', 'reenactor',
'breather', 'insensitive', 'hypersensitive', 'registry', 'scenery', 'poll',
'gravity', 'fruiterer', 'absence', 'astronomical', 'sermonize', 'problem',
'bootless', 'complimentary', 'placidity', 'scowl', 'fascinate',
'protectiveness', 'monk', 'preteens', 'disfavor', 'stroke', 'microcircuits',
'nomad', 'basketball', 'offensive', 'carburettors', 'largeness', 'preempt',
'fighting', 'supermarket', 'muscularity', 'perceptible', 'macroevolution',
'arab', 'flighted', 'reproachful', 'inducement', 'respectively', 'animalize',
'necessitate', 'keyboard', 'evidence', 'debarred', 'noble', 'sweetish',
'encrusted', 'submariners', 'negociate', 'initiate', 'rabbi', 'restrict',
'direction', 'blow', 'constancy', 'equality', 'defame', 'intending', 'swing',
'invitation', 'editor', 'madhouse', 'think', 'profit', 'possession',
'drownings', 'unfortunate', 'abandonment', 'virologist', 'aim', 'reasoning',
'radiance', 'excavations', 'remarkable', 'mechanical', 'accomplished',
'follower', 'prehistorical', 'important', 'insecurities', 'sponsor',
'difference', 'commode', 'pottery', 'skid', 'undefinable', 'monogram',
'circumvents', 'shortish', 'omission', 'icelandic', 'unintelligible',
'undiscerning', 'desire', 'fertility', 'change', 'satisfactory', 'reproves',
'authority', 'size', 'dissimulate', 'vaporise', 'surroundings', 'inflammation',
'eavesdropper', 'fever', 'postglacial', 'intercommunicate', 'self', 'possessor',
'infeasible', 'unclog', 'car', 'creative', 'retrace', 'managership',
'extension', 'extort', 'connect', 'lenience', 'embroideries', 'circumcising',
'secret', 'partner', 'convert', 'spoonfuls', 'cucumber', 'supposed',
'baptistic', 'intercession', 'categorization', 'immobilization', 'synchronized',
'structure', 'shackle', 'eye', 'science', 'impossible', 'delight', 'gladness',
'unsuitable', 'index', 'beat', 'sight', 'idiocy', 'underprivileged',
'companionships', 'unconscious', 'illiberal', 'attachment', 'radical',
'exclaiming', 'intelligences', 'unflagging', 'selectively', 'huffy',
'bastardize', 'misleading', 'benefited', 'noncitizens', 'reduce', 'prudery',
'extravert', 'toppled', 'postmodernist', 'adventism', 'eruptive', 'emulsify',
'incorrupt', 'subeditor', 'glass', 'liquid', 'discovery', 'player', 'drawers',
'unheralded', 'rustic', 'group', 'corrupt', 'feminised', 'maildrop',
'migrational', 'impermissible', 'predetermine', 'repel', 'relocation',
'enthusiastic', 'discountenance', 'company', 'engorge', 'religiousness',
'contravened', 'deadness', 'helm', 'transshipped', 'galvanic', 'acquisition',
'roosted', 'radiators', 'extraterrestrial', 'wild', 'hostility', 'asylum',
'perfectible', 'isosceles', 'composed', 'about', 'advancement', 'prophetical',
'transubstantiate', 'sentenced', 'unobjectionable', 'spiritualize',
'circumnavigations', 'encroachments', 'displeases', 'curvature', 'sexual',
'directional', 'approved', 'prisoners', 'law', 'designs', 'autobuses',
'blithering', 'career', 'therapeutical', 'label', 'trespass', 'crusaders',
'unmanned', 'unassertiveness', 'extinguish', 'integrity', 'internationaler',
'movie', 'contrarily', 'accordance', 'injure', 'antedating', 'spherical',
'fireproof', 'troops', 'acting', 'accessible', 'blessing', 'battleships',
'opinion', 'crier', 'flight', 'transfusing', 'professor', 'touch', 'expel',
'postmark', 'happiness', 'reasonable', 'organic', 'encapsulate', 'improving',
'unisexual', 'secretary', 'Arafat', 'amorphous', 'equip', 'interact', 'travel',
'cell', 'victim', 'conclusive', 'drink', 'speed', 'noisy', 'expressionless',
'series', 'subordination', 'animal', 'encouragement', 'bridge', 'entrance',
'exacted', 'knowing', 'fractionate', 'internationality', 'accommodation',
'image', 'nonfunctional', 'inmate', 'excitation', 'acrobat', 'hundred', 'motto',
'elated', 'dictatorship', 'evolution', 'withdrawal', 'functionality',
'corpulence', 'distributive', 'supernatural', 'depopulate', 'ship',
'monoculture', 'unquenchable', 'pathless', 'dangerous', 'rectorate',
'duplicable', 'literalness', 'run', 'bird', 'freakishly', 'surround',
'hospitalize', 'federalize', 'cheapen', 'tiger', 'conflagration', 'stimuli',
'comprehensive', 'inoffensive', 'ceaseless', 'militarize', 'designed',
'refinery', 'washers', 'disestablishing', 'preconception', 'resurfacing',
'spangle', 'homogenized', 'volunteer', 'agitation', 'board', 'careerism',
'farmer', 'undisputable', 'numerical', 'earmuffs', 'inscribe', 'virtuoso',
'respectable', 'besieging', 'imperils', 'know-how', 'partnership', 'clownish',
'differences', 'anticancer', 'assassinated', 'heterosexism', 'exterminator',
'reordering', 'unilateralist', 'utterance', 'goldplated', 'dematerialised',
'gelatinous', 'motivation', 'surpass', 'interesting', 'classify', 'fire', 'pay',
'homogeneous', 'cynically', 'scot', 'fabricate', 'topically', 'scope', 'film',
'unicyclist', 'monogenesis', 'device', 'postdates', 'infrastructure',
'authorship', 'immortalize', 'unsatisfactory', 'general', 'leverage', 'current',
'languishing', 'remitting', 'fight', 'insure', 'cocoon', 'ripeness', 'colonise',
'different', 'separationist', 'hop', 'election', 'connection', 'puritanism',
'academicism', 'demanded', 'guidance', 'bishop', 'carbonic', 'center',
'capture', 'traveler', 'outshout', 'unquestioned', 'creator', 'proton',
'quitter', 'jaguar', 'convector', 'disarranged', 'embroideress', 'disaster',
'septic', 'unexpected', 'property', 'infolding', 'galaxy', 'distinguishing',
'objectify', 'lusterware', 'enrollment', 'starkness', 'internet', 'splitter',
'supplanting', 'abnormality', 'deposit', 'valor', 'indoctrinate', 'grassroots',
'traversals', 'ordain', 'kilometer', 'rhymers', 'refurbishments', 'military',
'concert', 'merchandise', 'circumventing', 'execute', 'cofactors', 'transact',
'effectiveness', 'quadratics', 'agency', 'tournament', 'quicken', 'stoical',
'approachable', 'unrewarding', 'day', 'territory', 'confine', 'guest',
'sprinkle', 'inexpedient', 'regimental', 'undefeated', 'replications',
'obvious', 'elaborate', 'clozapine', 'museum', 'pave', 'depression', 'server',
'unintelligent', 'noise', 'black', 'slanderous', 'party', 'condition', 'focus',
'freshen', 'planet', 'hypertension', 'reduced', 'precociously', 'angrier',
'pitch', 'resides', 'cooperation', 'hospital', 'ostentatious', 'clericalism',
'christianise', 'attacker', 'deviationism', 'indicted', 'hilarity', 'invisible',
'fuck', 'lushness', 'commutation', 'deformity', 'tennis', 'unsalable',
'sportive', 'resistive', 'blunders', 'helical', 'dominance', 'urbanize',
'ecology', 'preposed', 'southern', 'rumbled', 'imposition', 'warning', 'mogul',
'divide', 'kindergarteners', 'lend', 'world', 'heater', 'edgeless', 'Jerusalem',
'aerialist', 'descend', 'internships', 'caramelize', 'protrusion', 'reckoner',
'inclosure', 'laud', 'rock', 'inconvertible', 'favourable', 'refuted',
'mistrustful', 'unmolested', 'transponder', 'critical', 'epicure', 'practice',
'harden', 'jarringly', 'case', 'Freud', 'seasonable', 'primer', 'long',
'predators', 'specialism', 'seriousness', 'uninformed', 'cynical', 'omnipotent',
'eat', 'improvement', 'fringes', 'nightly', 'inharmonious', 'inroad', 'popcorn',
'magically', 'convocation', 'domain', 'incalculable', 'hypercoaster',
'socialist', 'monograms', 'heedless', 'imitation', 'embody', 'brood',
'unilluminated', 'strengthened', 'muscle', 'stand-in', 'database',
'institutionalize', 'manner', 'star', 'disassembled', 'skidding', 'entity',
'marketers', 'papered', 'depreciate', 'reinterpret', 'unforgiving',
'horsemanship', 'Mars', 'friendship', 'interlingua', 'put', 'importance',
'conductance', 'attainment', 'labourer', 'evangelize', 'cardinality',
'consciousness', 'adapt', 'causing', 'knifing', 'impeded', 'indexical',
'unloved', 'classicist', 'kill', 'latinist', 'collection', 'expounded',
'microphallus', 'balance', 'marginalize', 'untroubled', 'interspecies',
'employments', 'classification', 'characterless', 'combination', 'salable',
'disfavoring', 'belief', 'encouraging', 'Yale', 'disassociates', 'unknowing',
'canvass', 'battened', 'acknowledgement', 'illiterate', 'personifying',
'recorders', 'cowboys', 'wine', 'baggers', 'naivety', 'gardens', 'dooming',
'forest', 'direct', 'discounters', 'sing', 'unskillfulness', 'cross-index',
'handbook', 'perfect', 'reclassifications', 'hypersensitivity', 'electrical',
'subserving', 'ruralist', 'regained', 'viewer', 'extraterrestrials',
'irremovable', 'transmitter', 'fuel', 'bounced', 'foreign', 'letter', 'buck',
'impotently', 'explorers', 'antitoxic', 'dig', 'growth', 'ruler', 'street',
'viscometry', 'contravene', 'victory', 'unblock', 'tail', 'romanic', 'wizard',
'intraspecific', 'moon', 'combusts', 'disbelieving', 'cofounders', 'earning',
'illegal', 'immobilizing', 'order', 'vindictiveness', 'become', 'quality',
'unicycles', 'halfhearted', 'demerit', 'exclamation', 'characteristic',
'politician', 'circumference', 'selling', 'fulfillments', 'nonpolitical',
'game', 'investor', 'sheepish', 'gloom', 'autocracy', 'partnerships',
'concreteness', 'antipsychotic', 'unzipping', 'usher', 'heartlessness',
'disorderly', 'rareness', 'cosponsoring', 'encoded', 'directionless',
'instruct', 'uncontrolled', 'nonconformist', 'discipline', 'yellowish',
'magnetic', 'subsurface', 'unbiased', 'requests', 'broadcasters', 'block',
'ticket', 'membership', 'weapon', 'sexism', 'tailgate', 'king', 'intersected',
'empty', 'hotel', 'uncommunicative', 'adaptive', 'actor', 'software',
'circumscribed', 'remember', 'transsexual', 'leadership', 'profitless',
'interest', 'interdisciplinary', 'primates', 'uninterested', 'abnormal',
'arouse', 'apolitical', 'spouse', 'sandwich', 'ennobled', 'concerti', 'enjoins',
'rooters', 'dissociable', 'acoustics', 'unreserved', 'dry', 'fantasist',
'frighten', 'presenting', 'discourteous', 'representational', 'fasten',
'parallelize', 'prejudge', 'cession', 'pledged', 'foresters', 'narrow-minded',
'distillate', 'preaching', 'censorships', 'seat', 'food', 'disengages',
'interlace', 'headless', 'sessions', 'subtropical', 'racket', 'reviewers',
'insidiously', 'spiritualist', 'banished', 'funeral', 'presence', 'profanity',
'strife', 'insurance', 'algebras', 'issue', 'football', 'unaffected', 'situate',
'imprecise', 'unprofessional', 'refresher', 'concurrency', 'incontestable',
'nerveless', 'worsens', 'regulate', 'legion', 'row', 'splice', 'behavioural',
'supplement', 'itch', 'transportation', 'letters', 'seeders', 'incubate',
'rhythmicity', 'theater', 'dissonance', 'prayer', 'racism', 'americanize',
'bellowing', 'populace', 'procurators', 'document', 'unwed', 'coeducation',
'cooperators', 'enhancement', 'skiing', 'moderatorship', 'inaccessible',
'disloyal', 'guardedly', 'adverse', 'finality', 'inheritable', 'increase',
'conjoins', 'secluding', 'hold', 'envelop', 'autograft', 'causative', 'smooth',
'equipment', 'uncertainty', 'provisionary', 'accommodative', 'microwaving',
'developments', 'disrespectful', 'reply', 'illimitable', 'circumvented', 'mind',
'commodes', 'astronautical', 'dependence', 'disk', 'confirmable', 'affirm',
'antechamber', 'guarantee', 'skilled', 'digitise', 'adjournment', 'contrive',
'marker', 'potent', 'postcode', 'vegetational', 'containers', 'undated',
'rehashing', 'perfective', 'approach', 'mingles', 'artlessness', 'government',
'currency', 'vindictively', 'royalist', 'unfavourable', 'postmodernism',
'memoir', 'founder', 'similarity', 'demoralise', 'flatulence', 'utilitarianism',
'binging', 'blurting', 'remounted', 'friendships', 'major', 'forecast', 'steep',
'analogous', 'unworthiness', 'duty', 'variation', 'interwove', 'nanosecond',
'confinement', 'urgency', 'radio', 'travelers', 'defrayed', 'outlawed',
'discriminatory', 'infectious', 'care', 'grinder', 'alarmism', 'extrajudicial',
'reproducible', 'analyze', 'talkativeness', 'command', 'extending', 'wealthy',
'circumspect', 'penitent', 'sprint', 'play', 'impolitic', 'fear', 'declare',
'synthesize', 'confluent', 'clergyman', 'italian', 'unacceptable', 'unsettled',
'percent', 'patrol', 'scattered', 'lubricate', 'robbery', 'educate', 'dark',
'hallucinating', 'guard', 'disavowed', 'unspecialised', 'interlaces', 'rub',
'hazard', 'longing', 'write', 'sit', 'ukrainians', 'censorship', 'intramural',
'love', 'Mexico', 'autobiographer', 'forbid', 'reinsured', 'music',
'revolutionise', 'humorous', 'incredulous', 'monarchical', 'gin', 'uproarious',
'reformism', 'ungraceful', 'pressurise', 'discordance', 'talk', 'freighter',
'victorious', 'corrode', 'tricolor', 'crisis', 'macroeconomist', 'publicise',
'dissenter', 'appraisal', 'large', 'coat', 'entertainer', 'merchantable',
'small', 'highlanders', 'bite', 'mathematician', 'retarding', 'posthole',
'playful', 'secondary', 'plant', 'abundance', 'enchantress', 'sufficed',
'untracked', 'predictive', 'undesirable', 'baste', 'queen', 'punctuate',
'children', 'expounding', 'observe', 'wealth', 'freshness', 'oil',
'championship', 'announcement', 'crispness', 'protraction', 'cliffhanger',
'interceptor', 'possess', 'postponements', 'eroticism', 'start', 'ejector',
'commit', 'listing', 'slack', 'snooper', 'autosuggestion', 'weaponize',
'figurative', 'magician', 'inquiring', 'impoliteness', 'emigration',
'acquisitive', 'mildness', 'thatcher', 'innocuous', 'anger', 'roofers', 'lease',
'consign', 'reputable', 'standing', 'hush', 'lithium', 'nonindulgent',
'harmful', 'semiconducting', 'practical', 'burying', 'environment', 'puffery',
'unloving', 'mutinied', 'prominence', 'microbiologist', 'criticism',
'enforcing', 'banquet', 'ear', 'interlinking', 'inheritances', 'paragraph',
'hydrochloride', 'characters', 'giving', 'authorize', 'spellers', 'syntactic',
'inquisitive', 'title', 'follow', 'greengrocery', 'conformism', 'insatiate',
'undemocratic', 'interpreter', 'immigrate', 'skateboarders', 'doctrine',
'unwanted', 'ascendence', 'zoo', 'advisory', 'dissenters', 'irritatingly',
'malevolence', 'believing', 'pleading', 'perceive', 'inbreeding',
'extraterritorial', 'irrationality', 'unfledged', 'unmarketable', 'atmosphere',
'shoot', 'carbonate', 'recycle', 'embellishment', 'wrathful', 'antifeminist',
'disguise', 'aid', 'psychodynamics', 'mother', 'brotherhood', 'philosophic',
'physically', 'collected', 'antitumor', 'postdated', 'broad', 'witch-hunt',
'move', 'unploughed', 'autobiographies', 'campfires', 'singing', 'ceramicist',
'self-discovery', 'telephone', 'object', 'emotional', 'inexplicable',
'brainless', 'vicarious', 'opportune', 'CD', 'symmetrical', 'organism',
'automobile', 'profusion', 'hover', 'link', 'rearrangements', 'kidnapped',
'skater', 'softness', 'depictive', 'down', 'regardless', 'term', 'exceedance',
'deviously', 'postposition', 'excitements', 'recast', 'denominate', 'sniffers',
'cosigns', 'amateurish', 'tea', 'dispossess', 'flightless', 'pestilence',
'assistance', 'preschooler', 'transposable', 'subfamily', 'diagonals',
'stickler', 'defensive', 'unwelcome', 'intermarry', 'shrewdness', 'regretful',
'monarchic', 'morality', 'fence', 'cofounder', 'resistor', 'excitedly',
'inquisitiveness', 'thing', 'spend', 'report', 'stay', 'remainder', 'lightship',
'perished', 'postholes', 'hypervelocity', 'enlist', 'investigator',
'condescend', 'asphaltic', 'incommensurate', 'imperfection', 'diver', 'inform',
'aspirate', 'interpreted', 'people', 'phenomenon', 'irrelevant', 'continence',
'gringo', 'cheat', 'computer', 'musical', 'graft', 'dam', 'extraversion',
'hypermarkets', 'midday', 'explain', 'drought', 'source', 'indiscriminate',
'unchaste', 'decorate', 'heraldist', 'exacerbated', 'prescriptions', 'physics',
'untrustworthy', 'plucked', 'tricolour', 'competition', 'promiscuous',
'devilish', 'impassively', 'speculate', 'preservation', 'reliable', 'entraps',
'lesson', 'amazings', 'assigned', 'connoting', 'sink', 'future', 'voice',
'hydrolysed', 'industry', 'titillated', 'tenured', 'pick', 'automate',
'medicate', 'indifferently', 'nonpublic', 'mccarthyism', 'ineffective',
'deserters', 'explorer', 'reviles', 'impulsion', 'potential', 'encamping',
'unarguable', 'interweaved', 'ringer', 'run-down', 'heiress', 'dishonest',
'hormone', 'clamorous', 'calculate', 'investigation', 'exempt', 'complain',
'demand', 'nonprofessional', 'devalue', 'support', 'price', 'reciprocal',
'holder', 'decay', 'monoclinic', 'sourdough', 'autoimmune', 'limit', 'pretense',
'hateful', 'separate', 'right', 'princedoms', 'significances', 'purposeless',
'give', 'intelligent', 'incongruous', 'proximity', 'wrongdoer', 'brandish',
'unfasten', 'auditive', 'preservers', 'medicine', 'suppleness', 'quieten',
'read', 'embroiderer', 'fieldworker', 'annoy', 'actuator', 'landscape',
'excrete', 'arbitrary', 'formations', 'suppress', 'aqueous', 'contest',
'inaccurate', 'indispensable', 'marathon', 'historically', 'subhead', 'trading',
'virility', 'insanity', 'inconsiderate', 'space', 'reformations',
'fragmentation', 'thick', 'contrastive', 'unsubdivided', 'inflection', 'code',
'interjection', 'obstruct', 'experimenter', 'observation', 'psychiatry',
'dimensional', 'boy', 'aluminum', 'separatist', 'psychologist', 'coupling',
'meaningless', 'londoners', 'grassland', 'rebel', 'nonrepresentational',
'century', 'Brazil', 'necessary', 'calmness', 'strangers', 'enlarger',
'predominance', 'hunt', 'meadows', 'intelligence', 'attempt', 'soloist',
'traverse', 'statistician', 'unequivocal', 'capitation', 'anticyclones', 'FBI',
'rubberstamp', 'confidence', 'mathematical', 'burn', 'frowning', 'adulteration',
'security', 'compartmentalization', 'rascality', 'vodka', 'together',
'aeronautical', 'antagonist', 'baseness', 'whizzed', 'construct',
'consequences', 'cosponsors', 'psychology', 'result', 'Israel', 'wisdom',
'piety', 'monoatomic', 'active', 'expressible', 'employable', 'industrialise',
'buy', 'bodily', 'cognizance', 'disprove', 'impartiality', 'unsexy',
'sternness', 'ravenous', 'sustainable', 'news', 'monocultures', 'carrier',
'machine', 'creation', 'lively', 'genre', 'lectureship', 'save', 'allergic',
'discipleship', 'internee', 'preserve', 'opalescence', 'hyperlink',
'decelerate', 'connectedness', 'disgruntle', 'cover', 'intracerebral',
'repositioned', 'repeating', 'reserve', 'subgroup', 'shanghai', 'autografts',
'protesters', 'nurturance', 'gravitated', 'tidings', 'interlinks', 'transfused',
'anaesthetics', 'misbehave', 'philanthropy', 'undissolved', 'crouch',
'hypothesis', 'venders', 'warrior', 'significant', 'wicked', 'sea',
'abbreviate', 'cemetery', 'past', 'implication', 'comport', 'settle',
'anarchist', 'proposition', 'autographic', 'chip', 'unionise', 'buggered',
'chooses', 'mimicked', 'coinsurance', 'convertible', 'conscripting',
'scheduled', 'corral', 'maker', 'sufferance', 'unicycle', 'presenters', 'allow',
'invite', 'arrangement', 'transfigure', 'confinements', 'frequency', 'voyage',
'control', 'angular', 'reprehensible', 'heartlessly', 'education', 'decoration',
'macroeconomists', 'refuels', 'reporters', 'point', 'practicality', 'empirical',
'breed', 'crystalline', 'unceremonious', 'hotness', 'sponge', 'please',
'trader', 'commingle', 'autoerotic', 'scarcity', 'mitigated', 'slacken',
'priest', 'encyclopaedic', 'assign', 'unmarried', 'team', 'deceiver',
'serenaded', 'interview', 'resigning', 'engineering', 'intramuscular', 'widen',
'circularize', 'dawn', 'stressor', 'homophobia', 'Jackson', 'short', 'validate',
'insecureness', 'consumer', 'energy', 'explicit', 'irrigate', 'astringe',
'smoothen', 'inanimate', 'delimitations', 'unconcerned', 'strength',
'pronunciation', 'wingless', 'credibility', 'loveless', 'coiled', 'immoveable',
'impurity', 'sensitivity', 'representable', 'discriminating', 'foreigners',
'intended', 'disabused', 'synoptic', 'lavishness', 'rite', 'baby', 'sailings',
'fleshiness', 'deconstruct', 'list', 'push', 'producing', 'assimilate',
'sidewinder', 'hardware', 'continuously', 'entreaty', 'probability',
'preordained', 'feline', 'wrongdoing', 'desertion', 'fixture', 'communicator',
'prideful', 'pious', 'normalise', 'cross-link', 'opposition', 'quarter',
'antonymous', 'skin', 'assessment', 'jewel', 'recovery', 'migrate',
'corespondent', 'office', 'discolor', 'cup', 'learn', 'administration', 'cost',
're-create', 'circle', 'status', 'doubt', 'credentials', 'communistic',
'rattlesnake', 'premise', 'request', 'enjoining', 'address', 'autofocus',
'refurbishment', 'clarify', 'palestinians', 'layer', 'state', 'ordinary',
'cuteness', 'roosters', 'foreigner', 'nanometer', 'negligence', 'airship',
'subspaces', 'drug', 'train', 'subdivided', 'librarianship', 'canker',
'knightly', 'momentousness', 'reprocessing', 'preliterate', 'omniscience',
'sincere', 'jazz', 'transfer', 'inch', 'helplessness', 'cognition', 'cosigned',
'shift', 'intensions', 'comfort', 'receiverships', 'schemer', 'indirect',
'contrabands', 'defeatist', 'exciting', 'molar', 'reproductive', 'treat',
'draw', 'comment', 'delay', 'ministry', 'generation', 'personnel', 'undefended',
'interrelationship', 'circumferential', 'tasty', 'intend', 'snookered',
'apologize', 'normalize', 'randomize', 'bunking', 'manslaughter', 'replication',
'consonant', 'fruit', 'bureaucrat', 'worker', 'high', 'fossil', 'territorials',
'microflora', 'intoxication', 'donate', 'blackmailed', 'polar', 'fingerprint',
'communication', 'departure', 'populate', 'waste', 'prospector', 'tripods',
'swan', 'inadvertence', 'string', 'iranian', 'ulcerate', 'cozy', 'architecture',
'early', 'balanced', 'record', 'out', 'improvise', 'lad', 'hankering', 'quote',
'enshrouded', 'call', 'purgatory', 'coerce', 'provincialism', 'pathfinder',
'contraries', 'disgorge', 'year', 'enforcements', 'established', 'associations',
'retrials', 'adhesion', 'involve', 'microbalance', 'political', 'irreligious',
'ceremonious', 'coefficient', 'asexual', 'tasteless', 'primacy', 'nonobservant',
'nerve', 'gibberish', 'interconnectedness', 'pathfinders', 'concerts', 'brown',
'airplane', 'register', 'maleness', 'self-discipline', 'shock', 'inference',
'dynastic', 'pilot', 'narrow-mindedness', 'untruth', 'install', 'monoplanes',
'wholeheartedness', 'faze', 'snickering', 'endangerment', 'closet',
'impermanent', 'benefactor', 'video', 'liability', 'chord', 'unapproachable',
'christianity', 'choke', 'mount', 'imply', 'acquiring', 'local',
'businessperson', 'conclusion', 'symbol', 'evil', 'oracle', 'diarrhea', 'bed',
'orientate', 'toxic', 'circumstances', 'gauge', 'reason', 'sex', 'reenact',
'absorbing', 'homoerotic', 'balminess', 'gathered', 'paper', 'deflowering',
'independently', 'irreverence', 'remakes', 'convergent', 'picket',
'electioneering', 'leisured', 'oppress', 'forceps', 'opera', 'constant',
'tolerable', 're-argue', 'department', 'persuasions', 'conscientious',
'communicativeness', 'morph', 'trilateral', 'loose', 'information',
'premeditation', 'spaciousness', 'emotionalism', 'rich', 'subdivide', 'bush',
'assistances', 'design', 'being', 'tend', 'library', 'suffer', 'lawyer',
'extractor', 'family', 'brittany', 'preheated', 'economic', 'hit', 'inelegance',
'attitude', 'kazakhstani', 'admit', 'dazzle', 'over', 'tricolours', 'summer',
'memorabilia', 'enroll', 'quarrel', 'passable', 'discontinuance', 'enliven',
'endorse', 'roller', 'deal', 'disembodied', 'tricycle', 'gain',
'incontrovertible', 'irresolution', 'ecclesiastic', 'operation', 'transvestite',
'canonize', 'brained', 'relates', 'tie', 'phosphate', 'successful', 'giant',
'paving', 'chairmanship', 'antifeminism', 'impossibilities', 'schnauzer',
'perspectives', 'exemplify', 'withhold', 'footballers', 'implement', 'tool',
'performing', 'incomprehension', 'sexless', 'antipsychotics', 'docile', 'knock',
'automates', 'population', 'combusted', 'dispersive', 'fiddled',
'intramolecular', 'commingled', 'spiciness', 'hard-and-fast', 'war',
'subroutines', 'preassembled', 'excommunicate', 'fly', 'locality', 'chemistry',
'bicycle', 'produce', 'trade', 'converse', 'hinduism', 'slaughterers',
'ashamed', 'discernment', 'soulfully', 'distinguish', 'removal', 'combustion',
'needleworker', 'portrayer', 'exclusive', 'relieve', 'territorial', 'bobbers',
'express', 'toss', 'whimsically', 'woman', 'plane', 'media', 'storm',
'researchers', 'physical', 'holy', 'adversely', 'teaspoonful', 'amusements',
'uninhibited', 'associate', 'greeting', 'juncture', 'embroiderers', 'vacations',
'agreement', 'agent', 'covered', 'individualist', 'concavity', 'wordless',
'immeasurable', 'hash', 'shout', 'noncivilized', 'determine', 'archive',
'legal', 'organization', 'garment', 'concerning', 'subsequences', 'subtend',
'spirited', 'workman', 'Palestinian', 'suggestible', 'condescended',
'conjurors', 'apply', 'accomplishments', 'flag', 'evangelistic', 'copulate',
'exist', 'separation', 'charge', 'boxing', 'formalisms', 'phone',
'objectifying', 'undatable', 'heading', 'monopolist', 'blend', 'photocopy',
'development', 'indelicate', 'immensely', 'filing', 'cylindric', 'sitting',
'writer', 'treatment', 'antagonize', 'inducted', 'loveable', 'shanked', 'plot',
'bench', 'magnetize', 'microfiche', 'decide', 'practicable', 'supply',
'intrude', 'place', 'compound', 'penetrate', 'acoustic', 'spiritless', 'stump',
'synthetical', 'break', 'bluejacket', 'clothes', 'periodical', 'winners',
'fording', 'advocate', 'range', 'unshaped', 'evangelicalism', 'transfuse',
'abstractionist', 'notebook', 'incurved', 'deity', 'same', 'instructorship',
'micrometer', 'rooster', 'parallelism', 'soccer', 'calcify', 'speculativeness',
'conformity', 'princedom', 'facilitation', 'discrete', 'admission',
'uncontroversial', 'irregardless', 'rampant', 'flim-flam', 'project',
'reelections', 'tenderize', 'craftsman', 'look', 'tripod', 'malfeasance',
'gaiety', 'up', 'unmentionable', 'florescence', 'contagious', 'sanctioned',
'destroy', 'regionalisms', 'honor', 'rededicated', 'strengthen', 'prompt',
'prolapse', 'extendible', 'expensiveness', 'work', 'circumnavigate',
'nominated', 'standardise', 'minister', 'formal', 'exterminated', 'angry',
'clinic', 'performer', 'wear', 'feudalism', 'calculation', 'hypermarket',
'potato', 'incorruptible', 'temptation', 'expert', 'amounted', 'outperforming',
'receiving', 'moisten', 'archery', 'expansion', 'satisfaction', 'meaning',
'remove', 'placement', 'convene', 'clients', 'receptions', 'liberation', 'wash',
'man', 'sprouting', 'clamor', 'apocalyptical', 'weather', 'entrapped',
'fictitiously', 'inapplicability', 'nazi', 'brother', 'hypertexts',
'sophisticate', 'standoffish', 'carefreeness', 'reconstructs', 'willingness',
'fashionable', 'defeat', 'article', 'encrust', 'tumble', 'OPEC', 'piquancy',
'unproductive', 'unite', 'marinate', 'teasingly', 'copilots', 'powerful',
'imperceptible', 'entwined', 'reclaim', 'virginals', 'providence', 'parameter',
'warmness', 'thunderstorm', 'masculinity', 'overlying', 'vanish', 'undress',
'sightedness', 'deface', 'disinflation', 'rediscovery', 'copying', 'category',
'top', 'apprenticeship', 'frivolously', 'customise', 'proceeding', 'concerto',
'comfortless', 'informative', 'drive', 'reporter', 'action', 'seize',
'vocalism', 'winking', 'criterion', 'type', 'uproariously', 'flood', 'round',
'exporters', 'reassuringly', 'congruity', 'economist', 'even', 'incoordination',
'transmutes', 'fighter', 'undisclosed', 'oppose', 'interviewing', 'comparing',
'logical', 'cultist', 'reorientate', 'matter', 'volatility', 'interstellar',
'psychic', 'gathering', 'riskless', 'disposition', 'sanskrit', 'bibliographies',
'cargo', 'planning', 'increasing', 'isolation', 'headship', 'essential',
'verbalize', 'suspenseful', 'covert', 'private', 'wreathe', 'competes',
'generalized', 'coast', 'unattainableness', 'discontinuous', 'embracement',
'detail', 'dollar', 'inquisitor', 'new', 'compatibility', 'slavic', 'voter',
'healthful', 'yodeling', 'reassessments', 'replaces', 'discoverys', 'church',
'ignorance', 'money', 'emulsifying', 'promised', 'interconnect', 'stamp',
'antisubmarine', 'constitutive', 'appearances', 'effected', 'dwarfish',
'scratch', 'relation', 'convict', 'relationship', 'destabilization',
'internationalisms', 'stimulation', 'slave', 'correspondence', 'eldership',
'macrocosmic', 'wanderers', 'microseconds', 'confide', 'objector',
'reservation', 'microorganism', 'amethysts', 'rompers', 'pregnancy', 'heavenly',
'allurement', 'excretion', 'discharged', 'inheritance', 'catalogued', 'brain',
'deprive', 'shore', 'reproduce', 'repress', 'relinquishment', 'colored',
'decompositions', 'cat', 'sensualist', 'prayerful', 'inflicted',
'preadolescent', 'thoughtless', 'absolute', 'greenly', 'repositions',
'nonverbally', 'baseball', 'trio', 'extrasensory', 'splashy', 'surprise',
'beverage', 'sorrowful', 'pour', 'arrival', 'analogize', 'newness', 'fast',
'number', 'elector', 'insertion', 'containership', 'unenthusiastic',
'absorbance', 'cuddle', 'experience', 'caliper', 'insubordinate', 'crosswise',
'interned', 'gluttonous', 'individual', 'preheating', 'dissatisfying',
'demureness', 'stock', 'density', 'disapproving', 'mayoralty', 'invariable',
'sulfide', 'scar', 'stormy', 'immoderate', 'surface', 'remarriage', 'caesarism',
'compose', 'admittance', 'registration', 'energized', 'chronologize',
'deployment', 'cowered', 'juvenile', 'considerable', 'challenge', 'compare',
'interrelate', 'plan', 'sterile', 'methodically', 'liverpools', 'originality',
'lordship', 'hilariously', 'aspect', 'attendance', 'deduce', 'regularize',
'beauty', 'remedy', 'extraordinary', 'unreal', 'person', 'seed', 'ablaze',
'chloride', 'graveyard', 'acceptable', 'scornful', 'worsen', 'form', 'thrust',
'manifestation', 'dead', 'idle', 'exploitive', 'spatiality', 'odorize',
'unconsciousness', 'bring', 'egg', 'globalise', 'imbedding', 'hypothetical',
'enfolding', 'bright', 'unfit', 'abashed', 'implantations', 'adjustor',
'optical', 'improver', 'senate', 'unforeseen', 'circumcision', 'supporters',
'syllable', 'propriety', 'young', 'excommunicated', 'democratize', 'gift',
'affordable', 'motion', 'insurrectionist', 'malicious', 'protractors',
'journal', 'belligerence', 'irrelevance', 'sanctify', 'seductive', 'ideality',
'punjabi', 'management', 'impregnate', 'predetermination', 'mercifulness',
'dependent', 'stove', 'flavourful', 'mark', 'airport', 'nonnative', 'curved',
'believe', 'observed', 'chemical', 'worthy', 'deregulating', 'mechanism',
'presuppose', 'unrealizable', 'disjoined', 'extrapolations', 'trilogies',
'subjugate', 'incised', 'wood', 'subarctic', 'Maradona', 'distrust',
'anterooms', 'mar', 'divided', 'transmuted', 'yen', 'smallish', 'inelasticity',
'distress', 'planners', 'proud', 'symbolist', 'parasitical', 'indecent',
'autoloading', 'griping', 'enunciated', 'trichloride', 'scholarships',
'flicker', 'part', 'dutch', 'card', 'nondescripts', 'act', 'awkward',
'transgress', 'portioned', 'decomposition', 'killer', 'revivalism', 'tense',
'inquirer', 'unconvincing', 'unvariedness', 'attackers', 'promotive',
'constrict', 'unwaveringly', 'comprehensible', 'affect', 'nonpartisan',
'composure', 'box', 'associational', 'clairvoyant', 'stimulates', 'corruptive',
'uncreative', 'irreproducible', 'locate', 'nobelist', 'ropewalker', 'cook',
'monarchist', 'brace', 'glistens', 'cooperator', 'retraced', 'laureate',
'reburial', 'lieutenant', 'conductive', 'repressing', 'protestantism',
'emergency', 'rematches', 'deceive', 'lustrate', 'discover', 'television',
'interchanging', 'dividend', 'institution', 'attractor', 'depressor',
'individualize', 'circumvolution', 'color-blind', 'acoustical', 'requirement',
'homophony', 'constellation', 'religion', 'braid', 'incomprehensible',
'sulfuric', 'pastorship', 'canonical', 'slurred', 'annihilator', 'concurrence',
'severer', 'swooshing', 'enthusiast', 'syphons', 'nominate', 'bestowals',
'safe', 'spoonful', 'personify', 'seven', 'transmissible', 'valorous',
'scarceness', 'sponsorship', 'microcomputers', 'procreating', 'continuous',
'antecedent', 'hear', 'decrease', 'disease', 'extroversive', 'laundering',
'subjoined', 'translocation', 'wallpapered', 'impartial', 'flaunt',
'sanctifying', 'monsignori', 'nosiness', 'equatorial', 'rarity', 'hungry',
'dull', 'followed', 'rid', 'procedure', 'isolate', 'written', 'sociability',
'purchasable', 'applaud', 'removes', 'oust', 'returning', 'regularise',
'nature', 'copartnership', 'prejudice', 'consultive', 'president', 'divisible',
'water', 'smart', 'postmarks', 'independent', 'chatter', 'commenting',
'totalism', 'strains', 'governor', 'nonstandard', 'grievous', 'fancy',
'lobster', 'roadless', 'expound', 'incombustible', 'country', 'soulless',
'procurator', 'spiritize', 'plague', 'pedicab', 'arousal', 'univocal',
'inflame', 'conditional', 'autoimmunity', 'float', 'autopilots', 'blacken',
'torment', 'postcodes', 'gumption', 'addiction', 'bohemia', 'postboxes',
'activity', 'crew', 'unambiguous', 'superficial', 'disturbance', 'disfigure',
'inarticulate', 'girl', 'raise', 'born', 'copilot', 'steal', 'luxury',
'exceptionally', 'hyperlinks', 'indian', 'pittance'}


# ### get the reduced embedding file

# In[50]:


def get_reduced_embedding(embed_path, output_path):
# filename = "reduced_embedding_4.txt"

    outfile = open(output_path, 'w')

    for row in open(embed_path):

        word, *_ = row.split()

        if word in words:
            outfile.write(row)
    outfile.close()


# In[ ]:


get_reduced_embedding("embedding_5.txt", "reduced_embedding_5.txt")


# ### test set similarity generation

# In[51]:



import sys
def test_similarity_geneartion(embed_path, output_path):
    def preprocess_word(s):
            s1 = s.lower()
            s1 = re.sub(RE_STRIP_SPECIAL_CHARS, '', s1)
            s1 = re.sub(RE_NUMBER, ' ', s1)
            s1 = s1.strip()
            if s1 in stop_words:
                s1 = ''
            else: 
                s1 = s1.strip()
            return s1
    def read_embedding(path):


        embedding = {}
        dim = None

        for row in open(path):

            word, *vector = row.split()
            try:
                embedding[word] = [float(x) for x in vector]
            except:
                continue

            if dim and len(vector) != dim:

                print("Inconsistent embedding dimensions!", file = sys.stderr)
                sys.exit(1)

            dim = len(vector)

        return embedding, dim


    E, dim = read_embedding(embed_path)
    pairs = pd.read_csv("data/similarity/test_x.csv", index_col = "id")
    cal_similarity = []
    for w1, w2 in zip(pairs.word1, pairs.word2):
        #process the w1,w2 like we did previously to the raw text
        w1 = preprocess_word(w1)
        w2 = preprocess_word(w2)

        if w1 not in E.keys():
            cur1 = '<UNK>'
        else:
            cur1 = w1
        if w2 not in E.keys():
            cur2 = '<UNK>'
        else:
            cur2 = w2
        cal_similarity.append(np.dot(E[cur1], E[cur2]))
    pairs["similarity"] = cal_similarity
    del pairs["word1"], pairs["word2"]
    pairs.to_csv(output_path)


# In[52]:


test_similarity_geneartion("embedding_5.txt", "results/test_prediction_5.csv")


# ## dealing with unknown words

# In[53]:



from argparse import ArgumentParser
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
nltk.download('wordnet')

def manual_stem(word):
    if word[-2:] == 'ly':
        return word[:-2]
    elif word[-3:] == 'ish':
        return word[:-3]
    else:
        return None


added_words = []
vocab = {}
unk_vector = ""

for row in open("embedding_3.txt"):

    word, *vec = row.split()
    vocab[word] = ' '.join(vec)

    if word in words:
        print(row, end = "")
        added_words.append(word)
    elif word == '<UNK>':
        unk_vector = row

counta = 0
countb = 0
countc = 0
countd = 0
counte = 0
countz = 0
ps = PorterStemmer()
filename = "modified_reduced_embedding_3.txt"

outfile = open(filename, 'w')



for w in words:
    if w not in added_words:
        vec = None
        # print(f'{w} not in vocab')
        found = False

        # find stemmed word in vocab
        stemmed_word = ps.stem(w)
        if stemmed_word in vocab:
            vec = w + ' ' + vocab[stemmed_word] + '\n'
            # print(f'!!found stemmed {word}')
            found = True
            countd += 1
        
        manual_stemmed_word = manual_stem(w)
        if manual_stemmed_word and manual_stemmed_word in vocab:
            vec = w + ' ' + vocab[manual_stemmed_word] + '\n'
            # print(f'!!found manually stemmed {manual_stemmed_word} for {w}')
            found = True
            countd += 1

        # find synonyms in vocab
        synonyms = wn.synsets(w)
        stemmed_synonyms = wn.synsets(stemmed_word)
        addl_stemmed_synonyms = []
        if manual_stemmed_word:
            addl_stemmed_synonyms = wn.synsets(manual_stemmed_word)
        synonyms += stemmed_synonyms + addl_stemmed_synonyms
        if not found:
            for synonym in synonyms:
                word = synonym.lemmas()[0].name()
                if word in vocab:
                    vec = w + ' ' + vocab[word] + '\n'
                    # print(f'!!found syn {word}')
                    found = True
                    counta += 1
                    break
        
        # find 1st level hyponyms of all synonyms
        if not found:
            for synonym in synonyms:    
                hyponyms = synonym.hyponyms()
                for hyponym in hyponyms:
                    hyponym_word = hyponym.lemmas()[0].name()
                    if hyponym_word in vocab:
                        vec = w + ' ' + vocab[hyponym_word] + '\n'
                        # print(f'!!found 1st level hyponym {hyponym_word}')
                        found = True
                        countb += 1
                        break
        
        # find 1st level hypernyms of all synonyms
        if not found:
            for synonym in synonyms:
                hypernyms = synonym.hypernyms()
                for hypernym in hypernyms:
                    hypernym_word = hypernym.lemmas()[0].name()
                    if hypernym_word in vocab:
                        vec = w + ' ' + vocab[hypernym_word] + '\n'
                        # print(f'!!found 1st level hypernym {hypernym_word}')
                        found = True
                        countc += 1
                        break
        
        # find all hyponyms recursively, for all synonyms
        if not found:
            for synonym in synonyms:
                all_hyponyms = [w for s in synonym.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]
                for hyponym_word in all_hyponyms:
                    stemmed_word = ps.stem(hyponym_word)
                    if hyponym_word in vocab:
                        vec = w + ' ' + vocab[hyponym_word] + '\n'
                        # print(f'!!found hyponym {hyponym_word} for {w}')
                        found = True
                        countb += 1
                        break
                    elif stemmed_word in vocab:
                        vec = w + ' ' + vocab[stemmed_word] + '\n'
                        # print(f'!!found hyponym {stemmed_word} for {w}')
                        found = True
                        countb += 1
                        break

        # find all hypernyms recursively, for all synonyms
        if not found:
            for synonym in synonyms:
                all_hypernyms = [w for s in synonym.closure(lambda s:s.hypernyms()) for w in s.lemma_names()]
                for hypernym_word in all_hypernyms:
                    stemmed_word = ps.stem(hypernym_word)
                    if hypernym_word in vocab:
                        vec = w + ' ' + vocab[hypernym_word] + '\n'
                        # print(f'!!found hypernym {hypernym_word} for {w}')
                        found = True
                        countc += 1
                        break 
                    elif stemmed_word in vocab:
                        vec = w + ' ' + vocab[stemmed_word] + '\n'
                        # print(f'!!found hypernym {stemmed_word} for {w}')
                        found = True
                        countc += 1
                        break
    
        if not found:
            for synonym in synonyms:
                antonyms = synonym.lemmas()[0].antonyms()
                for antonym in antonyms:
                    antonym_word = antonym.name()
                    stemmed_word = ps.stem(antonym_word)
                    if antonym_word in vocab:
                        vec = w + ' ' + vocab[antonym_word] + '\n'
                        found = True
                        countz += 1
                        break
                    elif stemmed_word in vocab:
                        vec = w + ' ' + vocab[stemmed_word] + '\n'
                        found = True
                        countz += 1
                        break
                        

        if not found:
            vec = w + unk_vector[5:] + ''
            counte += 1
        outfile.write(vec)
outfile.close()


# In[54]:



counta = 0
countb = 0
countc = 0
countd = 0
counte = 0
countz = 0


# In[94]:


def get_word_vector_wordnet(w, added_words,E,stemmedE, vocab):
#     vocab = V.keys()
    if w in added_words:
        return E[w]
    else:
        vec = None
        print(f'{w} not in vocab')
        found = False
        ps = PorterStemmer()
        # find stemmed word in vocab
        stemmed_word = ps.stem(w)
        if stemmed_word in vocab:
            vec = w + ' ' + vocab[stemmed_word] + '\n'
        
            print(f'!!found stemmed {w}')
            return E[stemmed_word]
            found = True
#             countd += 1
        
        manual_stemmed_word = manual_stem(w)
        if manual_stemmed_word and manual_stemmed_word in vocab:
            vec = w + ' ' + vocab[manual_stemmed_word] + '\n'
            print(f'!!found manually stemmed {manual_stemmed_word} for {w}')
            return E[manual_stemmed_word]
            found = True
#           countd += 1
        if stemmed_word in stemmedE.keys():
            return stemmedE[stemmed_word]

        # find synonyms in vocab
        synonyms = wn.synsets(w)
        stemmed_synonyms = wn.synsets(stemmed_word)
        addl_stemmed_synonyms = []
        if manual_stemmed_word:
            addl_stemmed_synonyms = wn.synsets(manual_stemmed_word)
        synonyms += stemmed_synonyms + addl_stemmed_synonyms
        if not found:
            for synonym in synonyms:
                word = synonym.lemmas()[0].name()
                if word in vocab:
                    vec = w + ' ' + vocab[word] + '\n'
                    print(f'!!found syn {word}')
                    return E[word]
                    found = True
#                     counta += 1
                    break
        
        # find 1st level hyponyms of all synonyms
        if not found:
            for synonym in synonyms:    
                hyponyms = synonym.hyponyms()
                for hyponym in hyponyms:
                    hyponym_word = hyponym.lemmas()[0].name()
                    if hyponym_word in vocab:
                        vec = w + ' ' + vocab[hyponym_word] + '\n'
                        print(f'!!found 1st level hyponym {hyponym_word}')
                        return E[hyponym_word]
                        found = True
#                         countb += 1
                        break
        
        # find 1st level hypernyms of all synonyms
        if not found:
            for synonym in synonyms:
                hypernyms = synonym.hypernyms()
                for hypernym in hypernyms:
                    hypernym_word = hypernym.lemmas()[0].name()
                    if hypernym_word in vocab:
                        vec = w + ' ' + vocab[hypernym_word] + '\n'
                        print(f'!!found 1st level hypernym {hypernym_word}')
                        return E[hypernym_word]
                        found = True
#                         countc += 1
                        break
        
        # find all hyponyms recursively, for all synonyms
        if not found:
            for synonym in synonyms:
                all_hyponyms = [w for s in synonym.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]
                for hyponym_word in all_hyponyms:
                    stemmed_word = ps.stem(hyponym_word)
                    if hyponym_word in vocab:
                        vec = w + ' ' + vocab[hyponym_word] + '\n'
                        print(f'!!found hyponym {hyponym_word} for {w}')
                        return E[hyponym_word]
                        found = True
#                         countb += 1
                        break
                    elif stemmed_word in vocab:
                        vec = w + ' ' + vocab[stemmed_word] + '\n'
                        print(f'!!found hyponym {stemmed_word} for {w}')
                        return E[stemmed_word]
                        found = True
#                         countb += 1
                        break

        # find all hypernyms recursively, for all synonyms
        if not found:
            for synonym in synonyms:
                all_hypernyms = [w for s in synonym.closure(lambda s:s.hypernyms()) for w in s.lemma_names()]
                for hypernym_word in all_hypernyms:
                    stemmed_word = ps.stem(hypernym_word)
                    if hypernym_word in vocab:
                        vec = w + ' ' + vocab[hypernym_word] + '\n'
                        print(f'!!found hypernym {hypernym_word} for {w}')
                        return E[hypernym_word]
                        found = True
#                         countc += 1
                        break 
                    elif stemmed_word in vocab:
                        vec = w + ' ' + vocab[stemmed_word] + '\n'
                        print(f'!!found hypernym {stemmed_word} for {w}')
                        return E[stemmed_word]
                        found = True
#                         countc += 1
                        break
    
        if not found:
            for synonym in synonyms:
                antonyms = synonym.lemmas()[0].antonyms()
                for antonym in antonyms:
                    antonym_word = antonym.name()
                    stemmed_word = ps.stem(antonym_word)
                    if antonym_word in vocab:
                        vec = w + ' ' + vocab[antonym_word] + '\n'
                        print(f'!!found antonym {antonym_word} for {w}')
                        return E[antonym_word]
                        found = True
#                         countz += 1
                        break
                    elif stemmed_word in vocab:
                        vec = w + ' ' + vocab[stemmed_word] + '\n'
                        print(f'!!found antonym {stemmed_word} for {w}')
                        return E[stemmed_word]
                        found = True
#                         countz += 1
                        break
                        

        if not found:
#             vec = w + unk_vector[5:] + ''
#             print(f'syn not found for {w}')
# #             counte += 1
            return E['<UNK>']
#         return [float(i) for i in vec.split()[1:]]
    
    
    
    


# In[95]:


def get_dev_similarity_wordnet(embed_path, output_path):
    def read_embedding(path):
        embedding = {}
        stemmed_embedding = {}
        dim = None
        for row in open(path):
            word, *vector = row.split()
#             try:
            embedding[word] = [float(x) for x in vector]
#             except:
#                 continue
            stemmed_word = ps.stem(word)
            stemmed_embedding[stemmed_word] = [float(x) for x in vector]

            if dim and len(vector) != dim:

                print("Inconsistent embedding dimensions!", file = sys.stderr)
                sys.exit(1)

            dim = len(vector)

        return embedding, stemmed_embedding,dim
    

    E, stemmedE, dim = read_embedding(embed_path)
    pairs = pd.read_csv("data/similarity/dev_x.csv", index_col = "id")
    #build the embedded word collection from the test set
    added_words = []
    vocab = {}
    unk_vector = ""
    words = set(pairs.word1).union(pairs.word2)
    for row in open(embed_path):

        word, *vec = row.split()
        vocab[word] = ' '.join(vec)

        if word in words:
#             print(row, end = "")
            added_words.append(word)
        elif word == '<UNK>':
            unk_vector = row

    

    cal_similarity = []
    for w1, w2 in zip(pairs.word1, pairs.word2):
        #first process
        w1 = preprocess_word(w1)
        w2 = preprocess_word(w2)
        #find the embedding, if the word not directly exist in the embedding, try to identify a synomon 
        cur1 = get_word_vector_wordnet(w1, added_words,E,stemmedE,vocab)
        cur2 = get_word_vector_wordnet(w2, added_words,E,stemmedE,vocab)
#         print(len(cur1)
#         print(cur2)
#         print(np.dot(np.array(cur1), np.array(cur2)))
        cal_similarity.append(float(np.dot(np.array(cur1), np.array(cur2))))
    pairs["similarity"] = cal_similarity
    
#     print(cal_similarity)
    # [np.dot(E[w1], E[w2])
    #     for w1, w2 in zip(pairs.word1, pairs.word2)]

    del pairs["word1"], pairs["word2"]

    # print("Detected a", dim, "dimension embedding.", file = sys.stderr)
    pairs.to_csv(output_path)
    


# In[80]:


# "lisztian" in vocab


# In[89]:


get_dev_similarity_wordnet("embedding_5.txt", "adjusted_dev_prediction_5.csv")


# In[90]:


evaluate_correlation("adjusted_dev_prediction_5.csv")


# ## test similarity generation with wordnet

# In[96]:


def get_test_similarity_wordnet(embed_path, output_path):
    def read_embedding(path):
        embedding = {}
        stemmed_embedding = {}
        dim = None
        for row in open(path):
            word, *vector = row.split()
#             try:
            embedding[word] = [float(x) for x in vector]
#             except:
#                 continue
            stemmed_word = ps.stem(word)
            stemmed_embedding[stemmed_word] = [float(x) for x in vector]

            if dim and len(vector) != dim:

                print("Inconsistent embedding dimensions!", file = sys.stderr)
                sys.exit(1)

            dim = len(vector)

        return embedding, stemmed_embedding,dim
    

    E, stemmedE, dim = read_embedding(embed_path)
    pairs = pd.read_csv("data/similarity/test_x.csv", index_col = "id")
    #build the embedded word collection from the test set
    added_words = []
    vocab = {}
    unk_vector = ""
    words = set(pairs.word1).union(pairs.word2)
    for row in open(embed_path):

        word, *vec = row.split()
        vocab[word] = ' '.join(vec)

        if word in words:
#             print(row, end = "")
            added_words.append(word)
        elif word == '<UNK>':
            unk_vector = row

    

    cal_similarity = []
    for w1, w2 in zip(pairs.word1, pairs.word2):
        #first process
        w1 = preprocess_word(w1)
        w2 = preprocess_word(w2)
        #find the embedding, if the word not directly exist in the embedding, try to identify a synomon 
        cur1 = get_word_vector_wordnet(w1, added_words,E,stemmedE,vocab)
        cur2 = get_word_vector_wordnet(w2, added_words,E,stemmedE,vocab)
#         print(len(cur1)
#         print(cur2)
#         print(np.dot(np.array(cur1), np.array(cur2)))
        cal_similarity.append(float(np.dot(np.array(cur1), np.array(cur2))))
    pairs["similarity"] = cal_similarity
    
#     print(cal_similarity)
    # [np.dot(E[w1], E[w2])
    #     for w1, w2 in zip(pairs.word1, pairs.word2)]

    del pairs["word1"], pairs["word2"]

    # print("Detected a", dim, "dimension embedding.", file = sys.stderr)
    pairs.to_csv(output_path)
    


# In[97]:


get_test_similarity_wordnet("embedding_5.txt", "results/adjusted_test_prediction_5.csv")
# test_similarity_geneartion("embedding_4.txt", "results/test_prediction_4.csv")


# In[100]:


def get_reduced_embedding_wordnet(embed_path, output_path):
    def read_embedding(path):
        embedding = {}
        stemmed_embedding = {}
        dim = None
        for row in open(path):
            word, *vector = row.split()
            embedding[word] = [float(x) for x in vector]

            stemmed_word = ps.stem(word)
            stemmed_embedding[stemmed_word] = [float(x) for x in vector]

            if dim and len(vector) != dim:

                print("Inconsistent embedding dimensions!", file = sys.stderr)
                sys.exit(1)

            dim = len(vector)

        return embedding, stemmed_embedding,dim
    

    E, stemmedE, dim = read_embedding(embed_path)
    added_words = []
    vocab = {}
    unk_vector = ""
    for row in open(embed_path):

        word, *vec = row.split()
        vocab[word] = ' '.join(vec)

        if word in words:
            added_words.append(word)
        elif word == '<UNK>':
            unk_vector = row
    outfile = open(output_path, 'w')
    for word in words:
        w1 = preprocess_word(word)
        cur1 = get_word_vector_wordnet(w1, added_words,E,stemmedE,vocab)
        temp = "".join(word.split()) + ' '
        for value in cur1:
            temp = temp + str(value) + ' '
        temp += '\n'
        outfile.write(temp)
        
    outfile.close()


# In[101]:


get_reduced_embedding_wordnet("embedding_4.txt", "adjusted_reduced_embedding_5.txt")




# In[ ]:




