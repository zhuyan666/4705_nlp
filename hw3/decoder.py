from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            input_rep = self.extractor.get_input_representation(words, pos, state)
            #print(list(self.model.predict(input_rep)[0]), max(list(self.model.predict(input_rep)[0])))
            probs = list(reversed(np.argsort(self.model.predict(input_rep)[0])))
            ind = 0
            for prob in probs:
                f = 0
                tran = self.output_labels[prob][0]
                if tran=='left_arc':
                    if len(state.stack)!=0 and (state.stack[-1]!=0):
                        f = 1
                elif tran=='right_arc':
                    if len(state.stack)!=0:
                        f = 1
                elif tran=='shift'
                    if (len(state.buffer)>1) or (len(state.buffer)==1 and len(state.stack)==0):
                        f = 1
                if f:
                    ind = prob
                    break
            action = self.output_labels[ind]
            if action[0] == 'shift':
                state.shift()
            if action[0] == 'left_arc':
                state.left_arc(action[1])
            if action[0] == 'right_arc':
                state.right_arc(action[1])
            # TODO: Write the body of this loop for part 4 

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in list(conll_reader(in_file))[:100]:
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
