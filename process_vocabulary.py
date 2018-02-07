import cPickle as pickle
import csv
import defs

def read_conll_file(fstream):
    """
    Reads a input stream @fstream (e.g. output of `open(fname, 'r')`) in CoNLL file format.
    @returns a list of examples [(tokens), (labels)]. @tokens and @labels are lists of string.
    """
    ret = []
    current_toks, current_lbls = [], []
    for line in fstream:
        line = line.strip()
        if len(line) == 0 or line.startswith("-DOCSTART-"):
            if len(current_toks) > 0:
                assert len(current_toks) == len(current_lbls)
                ret.append((current_toks, current_lbls))
            current_toks, current_lbls = [], []
        else:
            assert "\t" in line, r"Invalid CONLL format; expected a '\t' in {}".format(line)
            tok, lbl = line.split("\t")
            current_toks.append(tok)
            current_lbls.append(lbl)
    if len(current_toks) > 0:
        assert len(current_toks) == len(current_lbls)
        ret.append((current_toks, current_lbls))
    return ret


def casing(word):
    if len(word) == 0: return word
    # all lowercase
    if word.islower(): return "aa"
    # all uppercase
    elif word.isupper(): return "AA"
    # starts with capital
    elif word[0].isupper(): return "Aa"
    # has non-initial capital
    else: return "aA"


def words_to_ids(data,word_dict):
    # Preprocess data to construct an embedding
    if len(word_dict)==0:
       offset = 0
       print('word_dict offset = ',offset)
    else:
       offset = max(word_dict.values())

    for sentence, _ in data :
       for word in sentence :
           if word.isdigit(): word = defs.NUM
           else: word = word.lower()
           index = word_dict.setdefault(word,offset)
           offset = offset if index<offset else (offset+1)

    for i,word in enumerate([defs.P_CASE + c for c in defs.CASES],offset):
        word_dict.setdefault(word,i)
    
    offset = i+1
    for i,word in enumerate([defs.START_TOKEN, defs.END_TOKEN, defs.UNK],offset):
        word_dict.setdefault(word,i)

    sentences_ = []
    labels_ = []

    for sentence, label in data:
       #print('===========================')
       s = []
       k = 0
       for word in sentence:
           if word.isdigit(): word = defs.NUM
           else: word = word.lower()
           #print(word,word_dict.get(word,defs.UNK),label[k])
           #sentences_ += [[word_dict.get(word, word_dict[UNK]), word_dict[P_CASE + casing(word)]]]
           s += [word_dict.get(word, word_dict[defs.UNK])]
           k += 1
       sentences_ += [s]
       #for l in label:
       #    print(l,defs.LBLS.index(l))
       labels_ += [[defs.LBLS.index(l) for l in label]]

    #print(len(sentences_),sentences_)
    #print(len(labels_),labels_)
    return (sentences_,labels_)  
     

def make_windowed_data(data, start, end, window_size=1):

    # ensure data has both sentences and labels
    assert len(data)==2, 'data should be a tuple = (list of sentences, list of labels)'
    sentence_list = data[0] # sentence as a list of tokens e.g. [1,2,3,4]
    label_list = data[1]    # labels as a list of tokens   e.g. [0,1,1,4]

    orig_len = len(sentence_list)

    #extend the sentence_list with start and end tokens
    sentence_list = window_size*[start] + sentence_list + window_size*[end]

    output_list = []
    for i in range(window_size,window_size+orig_len):
      sentence_slice = sentence_list[i-window_size:i+window_size+1]
      label = label_list[i-window_size]
      tuple = (sentence_slice,label)
      output_list.append(tuple)

    #print 'windows = ',output_list
    return output_list

def to_string(s):
    # s = ([8711, 8711, 0, 1, 2], 4)
    return str(s).strip('()').strip('[').replace('],',';')

def process_sentences_and_labels(data,window_size,word_dict=None):
    if word_dict is None:
      word_dict = {}
    data = words_to_ids(data,word_dict)
    #print 'tokenized data : ',data 

    #start_token = [word_dict[START_TOKEN],word_dict[P_CASE + "aa"]]
    #end_token = [word_dict[END_TOKEN], word_dict[P_CASE + "aa"]]
    start_token = word_dict[defs.START_TOKEN]
    end_token = word_dict[defs.END_TOKEN]

    sentences_ = data[0] # list of tokenized sentences e.g. [[1,2,3,4],[5,6,7,8,9],[3,4,5],....]
    labels_ = data[1]    # list of tokenized labels    e.g. [[0,1,1,4],[0,0,3,3,4],[4,4,2],....]
    windowed_data = []
    for s,l in zip(sentences_,labels_):
       list_of_windows = make_windowed_data((s,l), start_token, end_token, window_size)
       # each window in list_of_windows is a tuple
       windowed_data += list_of_windows
    
    windowed_data_string = map(to_string,windowed_data)
    return (word_dict,windowed_data_string)

'''
# main
vocab_fstream = open('real_data/train.conll','r')
data = read_conll_file(vocab_fstream)
vocab_fstream.close()

word_dict,windowed_data = process_sentences_and_labels(data,defs.WINDOW_SIZE)

#print '\n\n' 
#print windowed_data 
#pickle.dump(windowed_data,open('windowed_data.p','wb'))

with open('real_data/train_windowed.csv', 'w') as f:
    writer = csv.writer(f, delimiter=';', lineterminator='\n')
    writer.writerows(windowed_data)

#dump the word to id dictionary also.
#inverted_dict = dict([[v,k] for k,v in word_dict.items()])

print(word_dict)
with open('real_data/train_word_dict.pkl','wb') as f:
   pickle.dump(word_dict,f)
'''
