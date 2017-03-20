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
from nltk import tokenize
import random
import itertools
from datetime import datetime
from itertools import islice, izip
from collections import Counter
from nltk.corpus import brown
from nltk.util import ngrams

TRAIN_DIR = os.path.join(os.getcwd(),'brown')
pattern = r'\w+:?(?=\/)'
bigrams = ngrams(brown.words(), 2)

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
        
        
        self.bigrams = Counter(bigrams)
        self.trans_prob = defaultdict(dict)
        self.emiss_prob = defaultdict(dict)
        self.posterior = defaultdict(dict)
        
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
        
    def sentence_prob(self,sentence):
        prob=0
        word_list = sentence.split()
        for word in word_list:
            word_prob= self.p_word_and_psuedocount(word)
            
            prob += math.log(word_prob)
           
        return prob
    
        
    def MLL(self):
        with open("Sentences.txt",'r') as doc:
            content = doc.read()
            sentences =tokenize.sent_tokenize(content)
        self.avg_log_prob += sum([self.sentence_prob(s) for s in sentences])
                
        print " Average log-likelihood per token : ", self.avg_log_prob
        print " Marginal log-likelihood per token : ", self.avg_log_prob/self.total_word_counts

        
    def e_step(self,K):
        """
        E-step of the EM algorithm to estmate posterior probability
        """
        
        
        for (z,word1,word2) in itertools.product(range(K),self.vocab,self.vocab):
            self.posterior[z][(word1,word2)]= self.trans_prob[z][word1]*self.emiss_prob[word2][z]/sum([self.trans_prob[l][word1]*self.emiss_prob[word2][l] for l in range(K)])
       
        
    def m_step(self,K,z,word1,emiss_den):
        """
        M-step of the algorithm to get transition and emission probabilities
        """
        
        trans_num=sum([self.bigrams[(x,y)]*self.posterior[z][(x,y)] for (x,y) in self.bigrams.keys() if x==word1])
        trans_den = sum( [ sum([ self.bigrams[(x,y)]*self.posterior[l][(x,y)] for (x,y) in self.bigrams.keys() if x==word1 ] ) for l in range(K)] )
        
        
        if trans_den == 0:
            self.trans_prob[z][word1]=(trans_num+1)/(trans_den + len(self.vocab))
        else:
            self.trans_prob[z][word1]= trans_num/trans_den 
   
       
        emiss_num=sum([self.bigrams[(x,y)]*self.posterior[z][(x,y)] for (x,y) in self.bigrams.keys() if y==word1])
        
        self.emiss_prob[word1][z]= emiss_num/emiss_den

       
        
    def emalgorithm(self,K,iterations):
        """
        Runs the EM algorithm over K latent classes for a specific number of iterations
        """
        
        print "Enter EM"
        emiss_den=[]
      
        self.trans_prob = {latent_class:{prev_word:random.random() for prev_word in self.vocab} for latent_class in range(K)}
                                       
        self.emiss_prob = {next_word:{latent_class:random.random() for latent_class in range(K)} for next_word in self.vocab}
                                    
        self.posterior={latent_class:{ (word1,word2):random.random() for (word1,word2) in self.bigrams.keys()} for latent_class in range(K)}  
        for z in range(K):
            emiss_den.append(sum([ self.bigrams[(word,word_bar)]*self.posterior[z][(word,word_bar)] for (word,word_bar) in self.bigrams.keys()]))                         
        print emiss_den
        while iterations > 0:
            iterations -= 1
            # E-step
            print "Running E-step for iteration :" ,iterations
            self.e_step(K)
            # M-step
            print "Running M-step for iteration :" ,iterations
            for (z,word1) in itertools.product(range(K),self.vocab):
                self.m_step(K,z,word1,emiss_den[z])
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
    lbm.MLL()
    #lbm.emalgorithm(3,1)
    print str(datetime.now())
    #sentence="colorless green ideas sleep furiously"
    #prob=lbm.log_prob_sentence(sentence,3)
   # print sentence,prob
       