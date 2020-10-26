import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Summer 2012 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    if n==1:
        seq = ['START']
    else:
        seq = ['START'] * (n-1)
    seq.extend(sequence)
    seq.append('STOP')

    n_grams = []
    for i in range(len(seq)-n+1):
        if isinstance(seq[i:i+n], list):
            n_grams.append(tuple(seq[i:i+n]))
        else:
            n_grams.append(tuple(list(seq[i:i+n])))

    return n_grams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Iterate through the corpus to count the number of words 
        generator = corpus_reader(corpusfile)
        self.n_words = sum([len(t) for t in generator])
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int) 
        self.trigramcounts = defaultdict(int) 

        unigrams, bigrams, trigrams = [], [], []
        for sentence in corpus:
            unigrams.append(get_ngrams(sentence, 1))
            bigrams.append(get_ngrams(sentence, 2))
            trigrams.append(get_ngrams(sentence, 3))
        unigrams = [item for sub in unigrams for item in sub]
        bigrams = [item for sub in bigrams for item in sub]
        trigrams = [item for sub in trigrams for item in sub]
        for i in unigrams:
            self.unigramcounts[i] += 1
        for i in bigrams:
            self.bigramcounts[i] += 1
        for i in trigrams:
            self.trigramcounts[i] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram[:2]==('START', 'START'):
            return self.trigramcounts[trigram] / self.unigramcounts[('START',)]

        if self.bigramcounts[trigram[:2]]==0:
            return 0
            
        return self.trigramcounts[trigram] / self.bigramcounts[trigram[:2]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        return self.bigramcounts[bigram] / self.unigramcounts[(bigram[0],)]

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return self.unigramcounts[unigram] / self.n_words

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        smoothed_trigram_prob = lambda1*self.raw_trigram_probability(trigram) + \
                                lambda2*self.raw_bigram_probability(trigram[1:]) + \
                                lambda3*self.raw_unigram_probability((trigram[2],))
        return smoothed_trigram_prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        log_prob = 0.
        for trigram in trigrams:
            log_prob += math.log2(self.smoothed_trigram_probability(trigram))

        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        l = 0
        M = 0
        for sentence in corpus:
            l += self.sentence_logprob(sentence)
            M += len(sentence)
        return 2**(-l/M)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        p1, p2 = [], []
        for f in os.listdir(testdir1):
            pp11 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp12 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp11<=pp12:
                correct += 1
            total += 1
    
        for f in os.listdir(testdir2):
            pp21 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp22 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if pp22<=pp21:
                correct += 1
            total += 1

        return correct / total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment('train_high.txt', "train_low.txt", "test_high", "test_low")                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    print(acc)

