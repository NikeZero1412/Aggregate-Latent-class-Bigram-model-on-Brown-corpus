# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 10:57:03 2017

@author: Nitin range(K)ishore

"""
from __future__ import division
import math
import os
from collections import defaultdict
import re
import sys
from nltk import tokenize
import random
import itertools
from datetime import datetime
from itertools import islice, izip
from collections import Counter
from nltk.corpus import brown
from nltk.util import ngrams
import numpy as np
from nltk import FreqDist
from sklearn.preprocessing import normalize


TRAIN_DIR = os.path.join(os.getcwd(),'brown')
pattern = r'\w+:?(?=\/)'
bigrams = ngrams(brown.words(), 2)
K=3
A[]
V=49742
sentences=[]
for sent in brown.sents()[:]:
    sent.insert(0, 'START')
    sent.append('END')
    sentences.append(sent)

def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return bow

    

class LatentClass_Bigram_Model:
    """Saul and Pereira’s “aggregate” latent-class bigram model"""

    def __init__(self):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()

        # class_total_doc_counts is a dictionary that maps a class to
        # the number of documents in the trainning set of that class
        self.total_sentence_counts = 0

        # class_total_word_counts is a dictionary that maps a class to
        # the number of words in the training set in documents of that class
        self.total_word_counts = 0
        self.avg_log_prob = 0
        self.word_counts = defaultdict(float)
        self.word_index=defaultdict(float)
        self.bigrams=FreqDist(bigrams)
        self.N_bigrams = np.asarray(self.bigrams.values())
        self.num_bigrams=len(self.bigrams)
        self.trans_prob = np.zeros((V,K))
        self.emiss_prob = np.zeros((V,K))
        self.posterior = np.zeros((self.num_bigrams,K))
        
        for i in range(len(self.vocab)):
            self.trans_prob[i] = np.random.dirichlet(np.ones(K),size=1)
            self.emiss_prob[i] = np.random.dirichlet(np.ones(K),size=1)
        
        for i in range(self.num_bigrams):
            self.posterior[i] = np.random.dirichlet(np.ones(K),size=1)
        print self.trans_prob.shape
        print self.emiss_prob.shape
        print self.posterior.shape
        
        
    def report_statistics_after_processing(self):
        """
        Report a number of statistics after processng.
        """

        print "REPORTING CORPUS STATISTICS"
        print "NUMBER OF SENTENCES IN CORPUS:", self.total_sentence_counts
        print "NUMBER OF TOKENS IN CORPUS:", self.total_word_counts
        print "VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab)
        print "NUMBER OF BIGRAMS IN CORPUS:", len(self.bigrams)
        
    def process_corpus(self,num_docs=None):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        
        """
      
        if num_docs is not None:
            print "Limiting to only %s docs " % num_docs
        filenames=os.listdir(TRAIN_DIR)
        if num_docs is not None: filenames = filenames[:num_docs]
        for f in filenames:
            with open(os.path.join(TRAIN_DIR,f),'r') as doc:
                content = doc.read()
                # Lower case and stripping POS tags
                processed = re.sub(r'/[^\s]+','',content)
                self.tokenize_and_update_model(processed)
        self.vocab.add("START")
        self.vocab.add("END")
        self.word_counts["START"] = self.total_sentence_counts
        self.word_counts["END"] = self.total_sentence_counts
        self.total_word_counts += self.word_counts["START"]+self.word_counts["END"]    
        self.word_index = dict((v, k) for k, v in enumerate(self.vocab))
        self.sum_tokens = self.posterior * self.N_bigrams[:, np.newaxis]        
        self.report_statistics_after_processing()
    
    def update_model(self, bow):
        """
        IMPLEMENT ME!

        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each class (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each class (self.class_total_doc_counts)
        """
        
        for k,v in bow.items():
            # for number of words we do +=1 and number of tokens we do +=v
            self.total_word_counts +=v
            self.word_counts[k]+=v
            self.vocab.add(k)    
            
    def tokenize_and_update_model(self, doc):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        Make sure when tokenizing to lower case all of the   tokens!
        """
        sentences =tokenize.sent_tokenize(doc)
        sentences= ["START "+ s.lower() + " END" for s in sentences ]
                    
        self.total_sentence_counts += len(sentences)
        
#        for s in sentences:
#            words = re.findall("\w+",s)
#            self.bigrams += Counter(izip(words, islice(words, 1, None)))
        bow = tokenize_doc(doc)
        self.update_model(bow)
            

                               
    def top_n(self, n):
        """
        Returns the most frequent n tokens for documents with class 'label'.
        """
        return sorted(self.word_counts.items(), key=lambda (w,c): -c)[:n]    

    def p_word_and_psuedocount(self,word):
        """
        Implement me!

        Returns the probability of word given label wrt psuedo counts.
        alpha - psuedocount parameter
        """
        V=len(self.vocab)
        total = self.total_word_counts
        
        count =self.word_counts[word]
        
        if word not in self.word_counts.keys() or self.word_counts[word]==0:
            alpha=1
        else:
            alpha=0
        prob_wgl= (count + alpha)/(total + (V*alpha))
        return prob_wgl
    
    def wordsprob(self,word1,word2):
        return np.sum(np.multiply(self.c_given_w1[self.word_index[word1.lower()]],self.w2_given_c[self.word_index[word2.lower()]]))
       

    def sentenceprob(self,sentence):
        prob=0
        for i in range(1,len(sentence)):
            if sentence[i] is not "END": 
                prob += math.log(self.wordsprob(sentence[i-1],sentence[i]))
            
        return prob
        
    def MLL(self):
        avg_log_prob=0
        avg_log_prob += sum([self.sentenceprob(s) for s in sentences])
                
        print " Average log-likelihood per token : ", avg_log_prob
        print " Marginal log-likelihood per token : ", avg_log_prob/self.num_tokens
        return avg_log_prob/self.num_tokens    
        
    def sentence_prob(self,sentence):
        prob=0
        word_list = sentence.split()
        for word in word_list:
            word_prob= self.p_word_and_psuedocount(word)
            
            prob += math.log(word_prob)
           
        return prob
    
        
    def MLL2(self):
        with open("Sentences.txt",'r') as doc:
            content = doc.read()
            sentences =tokenize.sent_tokenize(content)
        self.avg_log_prob += sum([self.sentence_prob(s) for s in sentences])
                
        print " Average log-likelihood per token : ", self.avg_log_prob
        print " Marginal log-likelihood per token : ", self.avg_log_prob/self.total_word_counts

        
    def e_step(self):
        """
        E-step of the EM algorithm to estmate posterior probability
        """
        index = 0
        for bigram in self.bigrams.keys():
            e_num = np.multiply(self.trans_prob[self.word_index[bigram[0]]], self.emiss_prob[self.word_index[bigram[1]]])
            self.posterior[index] = e_num/np.sum(e_num)
            index +=1
        
        
    def m_step(self):
        """
        M-step of the algorithm to get transition and emission probabilities
        """
        index=0
        
        m_num = self.posterior * self.N_bigrams[:, np.newaxis]
        for bigram in self.bigrams.keys():
            self.trans_prob[self.word_index[bigram[0]]] += m_num[index]
            self.emiss_prob[self.word_index[bigram[1]]] += m_num[index]
            index+=1
        self.trans_prob = normalize(self.trans_prob, norm='l1', axis=1)
        self.emiss_prob = normalize(self.emiss_prob, norm='l1', axis=0)
        
        self.sum_tokens = m_num
        print "Sum to tokens:",np.sum(self.sum_tokens.sum(axis=1) == self.N_bigrams)
        print "Sum of emission row probabilities :",self.emiss_prob.sum(axis=0)
        print "Sum of transition column probabilities :",self.trans_prob.sum(axis=1)
        
    def sent_ratio(self,sentence1,sentence2):
        
        logprob1= self.sentenceprob(sentence1.split())
        logprob2= self.sentenceprob(sentence2.split())
        ratio = logprob1/logprob2
        print "First sentence probability is :",logprob1
        print "Second sentence probability is :",logprob2
        print "Ratio is : ",ratio   
        
    def emalgorithm(self,iterations):
        """
        Runs the EM algorithm over K latent classes for a specific number of iterations
        """
        
        print "Enter EM"
        while iterations > 0:
            # E-step
            print "Running E-step for iteration :" ,iterations
            self.e_step()
            # M-step
            print "Running M-step for iteration :" ,iterations
            self.m_step()
            iterations -= 1

        print "EM complete"

               
    def prob_word1givenword2(self,word1,word2,K):
        K=[k for k in range(1,K+1)]
        return sum([self.trans_prob[z][word1]*self.emiss_prob[word2][z]for z in range(K)]) 
        
    def log_prob_sentence(self,sentence,K):
        K=[k for k in range(1,K+1)]
        f=sentence.split()
        prob=0
        for i in range(len(f)):
            prob += math.log(self.prob_word1givenword2(self,f[i],f[i+1],K))
        return prob
        
        
if __name__ == '__main__':
    lbm = LatentClass_Bigram_Model()
    lbm.process_corpus()
    print str(datetime.now())
    
    lbm.emalgorithm(1)
    A=lbm.MLL()
    print str(datetime.now())
    #sentence="colorless green ideas sleep furiously"
    #prob=lbm.log_prob_sentence(sentence,3)
   # print sentence,prob
    sentence1="*s* colorless green ideas sleep furiously *e*"
    sentence2="*s* furiously sleep ideas green colorless *e*"
    
    sentence3="*s* the u. s. government is paying for it . *e*"
    sentence4="*s* is the paying u. s. for government it . *e*"
    
    sentence5="*s* but jackie had gone into the station . *e*"
    sentence6="*s* into gone but had station jackie . *e*"
    
    sentence7="*s* but he warmed up after a while . *e*"
    sentence8="*s* he after but while up warmed . *e*"
    
    #plotting
    import matplotlib.pyplot as plt
    plt.plot(X, A, 'ro',X,B,'bs',X,C,'g^')
    plt.xlabel('Iteration')
    plt.ylabel('Marginal Log-Likelihood')
    plt.title('Variation of TokLL with EM iterations')
    plt.show()

       