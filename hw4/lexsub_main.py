#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

# added import 
import string

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    synsets = []
    l = wn.lemmas(lemma, pos=pos)
    for le in l:
        synsets.append(le.synset())
    lemmas = []
    for s in synsets:
        lemmas.extend(s.lemmas())
    ans = []
    for l in lemmas:
        ans.append(l.name())
    ans = [s for s in ans if s!=lemma]
    ans = [s.replace('_',' ') if '_' in s else s for s in ans ]

    return set(ans)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    lemma = context.lemma
    pos = context.pos
    l = wn.lemmas(lemma, pos=pos)
    synsets = [le.synset() for le in l]
    lemmas = []
    for s in synsets:
        lemmas.extend(s.lemmas())
    cnt = {}
    for l in lemmas:
        if l.name().lower() != lemma:
            name = l.name().replace('_', ' ')
            if name in cnt:
                cnt[name] += l.count()
            else:
                cnt[name] = l.count()
    #print(cnt)

    return max(cnt, key=cnt.get) # replace for part 2

def wn_simple_lesk_predictor(context : Context) -> str:
    stop_words = stopwords.words('english')
    # find synsets of the target word
    lemma = context.lemma
    pos = context.pos
    l = wn.lemmas(lemma, pos=pos)
    synsets = [le.synset() for le in l]
    # process the context of the target word
    target = context.left_context
    target.extend(context.right_context)
    target = [t.lower() for t in target if (t.lower() not in stop_words) and (t not in string.punctuation)]
    target = set(target)
    #print(target)
    # compute the overlap between the target and each of the synset
    ol_cnt = {}
    for s in synsets:
        # process the synset
        temp = list(s.definition())
        temp.extend(s.examples())
        hyper = []
        for h in s.hypernyms():
            hyper.extend(list(h.definition()))
            hyper.extend(h.examples())
        temp.extend(hyper)
        synset = []
        for t in temp:
            synset.extend(tokenize(t))
        synset = set([x.lower() for x in synset])
        # count the overlap
        c = 0
        for word in synset:
            if word in target:
                c += 1
        name = s.name().split('.')[0]
        ol_cnt[name] = c
    # if there's no overlap or there's tie
    l = list(ol_cnt.values())
    l.sort(reverse=True)
    if (sum(ol_cnt.values())==0): #or (l[0]>0 and l[1]>0 and l[0]==l[1]):
        t_cnt = {}
        for s in synsets:
            cnt = 0
            for l in s.lemmas():
                if l.name() == lemma:
                    cnt += l.count()
            t_cnt[s] = cnt
        best_s = max(t_cnt, key=t_cnt.get)
        l_cnt = {}
        for l in best_s.lemmas():
            l_cnt[l.key()] = l.count()
        best_l = max(l_cnt, key=l_cnt.get).split('%')[0]
        if best_l == lemma:
            return 'smurf'
        else:
            return best_l.replace('_', ' ')
    if len(l) > 1:
        if (l[0]>0 and l[1]>0 and l[0]==l[1]):
            t_cnt = {}
            for s in synsets:
                cnt = 0
                for l in s.lemmas():
                    if l.name() == lemma:
                        cnt += l.count()
                t_cnt[s] = cnt
            best_s = max(t_cnt, key=t_cnt.get)
            l_cnt = {}
            for l in best_s.lemmas():
                l_cnt[l.key()] = l.count()
            best_l = max(l_cnt, key=l_cnt.get).split('%')[0]
            if best_l == lemma:
                return 'smurf'
            else:
                return best_l.replace('_', ' ')     


    return max(ol_cnt, key=ol_cnt.get).replace('_', ' ') #replace for part 3        
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        max_sim = 0
        candidates = get_candidates(context.lemma, context.pos)
        for c in candidates:
            f = 1
            if c in self.model.wv:
                f = 0
                sim = self.model.similarity(context.lemma, c)
                if sim > max_sim:
                    max_sim = sim
                    best_syn = c
        if f:
            return 'smurf'

        return best_syn.replace('_', ' ') # replace for part 4


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        inp = context.left_context
        inp.append('[MASK]')
        inp.extend(context.right_context)
        mask_ind = inp.index('[MASK]')
        inp = ' '.join(inp)
        input_toks = self.tokenizer.encode(inp)
        mask_ind = self.tokenizer.convert_ids_to_tokens(input_toks).index('[MASK]')
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][mask_ind])[::-1]
        best_words = self.tokenizer.convert_ids_to_tokens(best_words)
        for word in best_words:
          if word in candidates:
             return word.replace('_', ' ')

        return 'smurf'      
        # replace for part 5


# part 6
def p6(context: Context) -> str:
    l = wn.lemmas(context.lemma, pos=context.pos)
    synsets = [le.synset() for le in l if len(le.synset().lemmas())>1]
    cnt = {}
    for s in synsets:
      c = 0
      for l in s.lemmas():
        c += l.count()
      name = s.name().split('.')[0].replace('_', ' ')
      cnt[s] = c
    best_s = max(cnt, key=cnt.get)
    max_c = 0
    for l in best_s.lemmas():
      if l.name() != context.lemma:
        if l.count() >= max_c:
          max_c = l.count()
          best_l = l.name().replace('_', ' ')
    
    return best_l


    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    predictor = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        #prediction = smurf_predictor(context) 
        #prediction = wn_frequency_predictor(context)
        #prediction = wn_simple_lesk_predictor(context)
        #prediction = predictor.predict_nearest(context)
        prediction = predictor.predict(context)
        #prediction = p6(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
        