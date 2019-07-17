import time
start=time.time()
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import CategorizedPlaintextCorpusReader
from sklearn.cluster import KMeans
import numpy as np
import copy
import math
import re

corpus_root = 'C:\\MyData\\PythonPractice\\IMDB\\train' #Path of IMDB Train Data
reader=CategorizedPlaintextCorpusReader(corpus_root,r'.*\.txt',cat_pattern=r'(\w+)/*')

r_pos=reader.fileids(categories=['pos'])
r_neg=reader.fileids(categories=['neg'])

global_shortlisted=[]
TRAIN_GS_POS=[]

for i in range(0,12500):
    
    doc=reader.raw(r_pos[i:i+1])   #doc contains the movie review
    sentences = nltk.sent_tokenize(doc)
    senlen=len(sentences)
    
    def decontracted(phrase):
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
    
        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase
    
    for k in range(senlen):
        sentences[k]=decontracted(sentences[k])
    
    stokens = [nltk.word_tokenize(sent) for sent in sentences]
    taggedlist=[]
    
    for stoken in stokens:        
         taggedlist.append(nltk.pos_tag(stoken))
    
    wnl = nltk.WordNetLemmatizer()
        
    #CALCULATION OF FUZZY SCORES
    score_list=[]
    score_list_pos=[]
    score_list_neg=[]
    global_swnwords_list=[]
    
    for idx,taggedsent in enumerate(taggedlist):
        score_list.append([])
        score_list_pos.append([])
        score_list_neg.append([])
        global_swnwords_list.append([])
        
    # Nouns,Adjectives,Verbs,Adverbs
        for idx2,t in enumerate(taggedsent):
            newtag=''
            lemmatized=wnl.lemmatize(t[0])
            if t[1].startswith('NN'):
                newtag='n'
            elif t[1].startswith('JJ'):
                newtag='a'
            elif t[1].startswith('V'):
                newtag='v'
            elif t[1].startswith('R'):
                newtag='r'
            else:
                newtag=''       
            if(newtag!=''):    
                synsets = list(swn.senti_synsets(lemmatized, newtag)) # Use of SentiWordNet Lexicon
        
                if(synsets): 
                    global_swnwords_list[idx].append(lemmatized)
                
                score=0
                scorepos=scoreneg=0
                if(len(synsets)>0):
                    for syn in synsets:
                        
                        scorepos+=syn.pos_score()
                        scoreneg+=syn.neg_score()
                        score+=syn.pos_score()-syn.neg_score()
                        
                    score_list_pos[idx].append(round((scorepos/len(synsets)),4))
                    score_list_neg[idx].append(round((scoreneg/len(synsets)),4))
                    score_list[idx].append(round((score/len(synsets)),4))
    
#    print(global_swnwords_list)
    global_entropy=[]
    mid=0
    
    #CALCULATION OF FUZZY ENTROPY SCORES
    
    for j in range(senlen):  # Execution of each sentence
        
        SentimentScore=[]
        d=[]        
        SentimentScore.append(score_list[j])
        d=copy.copy(SentimentScore[0])  #Sentiment Score (µd) for each word in sentence
        
        fuzzy=[]
        entropy=[]
        sentivalue=[]
        
        for k in range(len(d)):
            
            if (max(d)-min(d))==0:
                m=0
                fuzzy.append(0.0)
            else: 
                m= (d[k]-min(d))/(max(d)-min(d))
                fuzzy.append(round(m,4))       
            eps=pow(10,-12)
            ent=-(m*math.log(m+eps))-((1-m)*math.log((1-m)+eps))
            temp_ent=round(ent,4)
            if temp_ent==-0.0:
                temp_ent=0.0
            entropy.append(temp_ent)
            global_entropy.append(temp_ent)
    
    #USING k-means CLUSTERING ALGORITHM TO SELECT THE THRESHOLD FOR ENTROPY
            
    x1=np.asarray(global_entropy)  
    len(x1)
    t=len(x1)+1
    x2 = np.array([*range(1,t)])
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    # KMeans algorithm 
    K = 2
    kmeans=KMeans(n_clusters=K)
    kmeans_model = kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    c=centroids[0]
    d=centroids[1]
    th=round(c[0],4)
    th2=round(d[0],4)
    mid=round(((th+th2)/2),4)
#    print("\n Global Midpoint Centroid is :"+str(mid)) # THRESHOLD
    
    
    #CALCULATION OF FUZZY ENTROPY SCORES AND SELECTION OF FUZZY ENTROPY SCORES USING THRESHOLD
      
    para_shortlisted=[]      
    for j in range(senlen): #Execution of each sentence
        
#        print("\n" + str(j+1)+ ": "+ sentences[j])   
            
        b=[]
        b.append(global_swnwords_list[j])
        swntagwords=[]        
        swntagwords=copy.copy(b[0])

        SentimentScore=[]
        SentimentScore.append(score_list[j])
        d=[]       
        d=copy.copy(SentimentScore[0])
    
        check_ent=[]  # list to store indexes of selected entropy words
        selected_scores=[] #selected entropy values
        
        fuzzy=[]
        entropy=[]
        
        for k in range(len(d)):
            
            if (max(d)-min(d))==0:
                m=0
                fuzzy.append(0.0)
            else: 
                m= (d[k]-min(d))/(max(d)-min(d))
                fuzzy.append(round(m,4))       
            eps=pow(10,-12)
            ent=-(m*math.log(m+eps))-((1-m)*math.log((1-m)+eps))
            temp_ent=round(ent,4)
            if temp_ent==-0.0:
                temp_ent=0.0
            entropy.append(temp_ent)
            global_entropy.append(temp_ent)
              
            if entropy[k]<mid:     # Fuzzy Entropy Scores less than Threshold are selected
                t=k
                check_ent.append(t)
    
        ss=[]
        ss.append([swntagwords[a] for a in check_ent])
        swn_shortlisted=[]                      # Shorlisted words
        swn_shortlisted=copy.copy(ss[0])
        
        swnlen=len(swn_shortlisted)  # number of shortlisted words for each sentence
    
        para_shortlisted.append(swn_shortlisted)
            
#    print("\n Shortlisted words in a review: "+ str(para_shortlisted))
    global_shortlisted.append(para_shortlisted)
            
print("\n Global Shortlisted words in training: "+ str(global_shortlisted))

TRAIN_GS_POS=global_shortlisted

global_shortlisted2=[]
TEST_GS_POS=[]

for i in range(0,12500):
    
    doc=reader.raw(r_neg[i:i+1])
    sentences = nltk.sent_tokenize(doc)
    senlen=len(sentences)
    
    def decontracted(phrase):
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
    
        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase
    
    for k in range(senlen):
        sentences[k]=decontracted(sentences[k])
    
    stokens = [nltk.word_tokenize(sent) for sent in sentences]
    taggedlist=[]
    
    for stoken in stokens:        
         taggedlist.append(nltk.pos_tag(stoken))
    
    wnl = nltk.WordNetLemmatizer()
        
    #CALCULATION OF FUZZY SCORES
    score_list=[]
    score_list_pos=[]
    score_list_neg=[]
    global_swnwords_list=[]
    
    for idx,taggedsent in enumerate(taggedlist):
        score_list.append([])
        score_list_pos.append([])
        score_list_neg.append([])
        global_swnwords_list.append([])
        
    # Nouns,Adjectives,Verbs,Adverbs
        for idx2,t in enumerate(taggedsent):
            newtag=''
            lemmatized=wnl.lemmatize(t[0])
            if t[1].startswith('NN'):
                newtag='n'
            elif t[1].startswith('JJ'):
                newtag='a'
            elif t[1].startswith('V'):
                newtag='v'
            elif t[1].startswith('R'):
                newtag='r'
            else:
                newtag=''       
            if(newtag!=''):    
                synsets = list(swn.senti_synsets(lemmatized, newtag)) # Use of SentiWordNet Lexicon
        
                if(synsets): 
                    global_swnwords_list[idx].append(lemmatized)
                
                score=0
                scorepos=scoreneg=0
                if(len(synsets)>0):
                    for syn in synsets:
                        
                        scorepos+=syn.pos_score()
                        scoreneg+=syn.neg_score()
                        score+=syn.pos_score()-syn.neg_score()
                        
                    score_list_pos[idx].append(round((scorepos/len(synsets)),4))
                    score_list_neg[idx].append(round((scoreneg/len(synsets)),4))
                    score_list[idx].append(round((score/len(synsets)),4))
    
#    print(global_swnwords_list)
    global_entropy=[]
    mid=0
    
    for j in range(senlen):  # Execution of each sentence
        
        SentimentScore=[]
        d=[]        
        SentimentScore.append(score_list[j])
        d=copy.copy(SentimentScore[0])  #Sentiment Score (µd) for each word in sentence
        
        fuzzy=[]
        entropy=[]
        sentivalue=[]
        
        for k in range(len(d)):
            
            if (max(d)-min(d))==0:
                m=0
                fuzzy.append(0.0)
            else: 
                m= (d[k]-min(d))/(max(d)-min(d))
                fuzzy.append(round(m,4))       
            eps=pow(10,-12)
            ent=-(m*math.log(m+eps))-((1-m)*math.log((1-m)+eps))
            temp_ent=round(ent,4)
            if temp_ent==-0.0:
                temp_ent=0.0
            entropy.append(temp_ent)
            global_entropy.append(temp_ent)
    
    #USING k-means CLUSTERING ALGORITHM TO SELECT THE THRESHOLD FOR ENTROPY
            
    x1=np.asarray(global_entropy)  
    len(x1)
    t=len(x1)+1
    x2 = np.array([*range(1,t)])
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    # KMeans algorithm 
    K = 2
    kmeans=KMeans(n_clusters=K)
    kmeans_model = kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    c=centroids[0]
    d=centroids[1]
    th=round(c[0],4)
    th2=round(d[0],4)
    mid=round(((th+th2)/2),4)
#    print("\n Global Midpoint Centroid is :"+str(mid)) # THRESHOLD
     
    #CALCULATION OF FUZZY ENTROPY SCORES AND SELECTION OF FUZZY ENTROPY SCORES USING THRESHOLD

    para_shortlisted=[]      
    for j in range(senlen): # Execution of each sentence
        
#        print("\n" + str(j+1)+ ": "+ sentences[j])   
            
        b=[]
        b.append(global_swnwords_list[j])
        swntagwords=[]        
        swntagwords=copy.copy(b[0])

        SentimentScore=[]
        SentimentScore.append(score_list[j])
        d=[]       
        d=copy.copy(SentimentScore[0])
    
        check_ent=[]  # list to store indexes of selected entropy words
        selected_scores=[] #selected entropy values
        
        fuzzy=[]
        entropy=[]
        
        for k in range(len(d)):
            
            if (max(d)-min(d))==0:
                m=0
                fuzzy.append(0.0)
            else: 
                m= (d[k]-min(d))/(max(d)-min(d))
                fuzzy.append(round(m,4))       
            eps=pow(10,-12)
            ent=-(m*math.log(m+eps))-((1-m)*math.log((1-m)+eps))
            temp_ent=round(ent,4)
            if temp_ent==-0.0:
                temp_ent=0.0
            entropy.append(temp_ent)
            global_entropy.append(temp_ent)
              
            if entropy[k]<mid:   # Fuzzy Entropy Scores less than Threshold are selected
                t=k
                check_ent.append(t)
    
        ss=[]
        ss.append([swntagwords[a] for a in check_ent])
        swn_shortlisted=[]                      # Shorlisted words
        swn_shortlisted=copy.copy(ss[0])
        
        swnlen=len(swn_shortlisted)  # number of shortlisted words for each sentence
    
        para_shortlisted.append(swn_shortlisted)
            
#    print("\n Shortlisted words in a review: "+ str(para_shortlisted))
    global_shortlisted2.append(para_shortlisted)
            
print("\n Global Shortlisted words in training: "+ str(global_shortlisted2))

TRAIN_GS_NEG=global_shortlisted2

import numpy as np
from keras.datasets import imdb
vocabulary_size = 5000

word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}

def rearrange_para(alist):
    newlist=[]
    for i in range(len(alist)):
        s=alist[i]
        for j in range(len(s)):
            t=s[j]
            newlist.append(t)
    return(newlist)
    
def rearrange_whole(alist):
    nlist=[]
    for i in range(len(alist)):
        t=rearrange_para(alist[i])
        nlist.append(t)
    return(nlist)
            
 
Train_SH_P_W=rearrange_whole(TRAIN_GS_POS)
Train_SH_N_W=rearrange_whole(TRAIN_GS_NEG)

def calc_wordtonum_sen(wordlist): # Gives list of numbers for each word in sentence
    
    numlist=[]
    for i in range(len(wordlist)):
        if (wordlist[i].lower() in word2id):
            t=word2id[wordlist[i].lower()]
            if(t<5000):
                numlist.append(t)
    return(numlist)      
    
def calc_wordtonum_para(parawordlist): # Gives list of numbers for each word in sentence, for all sentences in paragraph
   
    numlist=[]
    for i in range(len(parawordlist)):
        t=parawordlist[i]
        f=calc_wordtonum_sen(t)
        numlist.append(f)  
    return(numlist)    
    
SH_POS_Train=calc_wordtonum_para(Train_SH_P_W)
SH_NEG_Train=calc_wordtonum_para(Train_SH_N_W)

np_pos_train=np.ones((len(SH_POS_Train),), dtype=int)
np_neg_train=np.zeros((len(SH_NEG_Train),), dtype=int)

Train=[]   #Contains all the training data of positive and negative reviews
for i in range(len(SH_POS_Train)):
    Train.append(SH_POS_Train[i])
for j in range(len(SH_NEG_Train)):
    Train.append(SH_NEG_Train[j])
#print(Train)

Train_labels=[]  #Contains the sentiment labels for all training data of positive and negative reviews
for i in range(len(np_pos_train)):
    Train_labels.append(np_pos_train[i])
for j in range(len(np_neg_train)):
    Train_labels.append(np_neg_train[j])
#print(Train_labels)

import os
import numpy as np

BASE_PATH = "C:\\MyData\\PythonPractice\\npzfiles"

file_name = "trainfile.npz".format(i)
final_name=os.path.join(BASE_PATH, file_name)
np.savez(final_name,Trainset=Train,TrainLabels=Train_labels)

end=time.time()
print(str(end-start)+"secs")