import nltk, superchunk_reader, re, collections
from nltk.metrics import precision, recall
from nltk.classify import accuracy
from nltk.corpus.reader import *
from nltk.tokenize import RegexpTokenizer

'''Setup for training/test data'''
training_files = '/Users/jasonurton/Desktop/PA4_Data/training'
test_files = '/Users/jasonurton/Desktop/PA4_Data/test'

training_reader = ChunkedCorpusReader(training_files, r'wsj_.*\.pos',
    sent_tokenizer=RegexpTokenizer(r'(?<=/\.)\s*(?![^\[]*\])', gaps=True),
    para_block_reader=tagged_treebank_para_block_reader,
    str2chunktree=superchunk_reader.superchunk2tree)
    
test_reader = ChunkedCorpusReader(test_files, r'wsj_.*\.pos',
    sent_tokenizer=RegexpTokenizer(r'(?<=/\.)\s*(?![^\[]*\])', gaps=True),
    para_block_reader=tagged_treebank_para_block_reader,
    str2chunktree=superchunk_reader.superchunk2tree) 

training_set = []; test_set = []

for file in training_reader.chunked_paras():
    for para in file:
        training_set.append(list(superchunk_reader.tree2iob(para)))

for file in test_reader.chunked_paras():
    for para in file:
        test_set.append(list(superchunk_reader.tree2iob(para)))

'''Feature generation methods'''
def get_previous_word(sent, i):
    if i == 0: return '<START>'
    else: return sent[i-1][0]

def get_SNP_suffix_1(sent, i):
    if i == 0: return '<START>'
    else: return sent[i-1][3]
        
def get_SNP_suffix_2(sent, i):
    if i <= 1: return '<START>'
    else: return sent[i-2][3]

def get_features(word, sent, i):
    features = {}
    features['Current Word'] = word[0]
    features['POS'] = word[1]
    features['NP Tag'] = word[2]
    features['Previous Word'] = get_previous_word(sent, i)
    features['SNP Suffix (-1)'] = get_SNP_suffix_1(sent, i)
    features['SNP Suffix (-2)'] = get_SNP_suffix_2(sent, i)
    return features

def get_featureset(data, featureset = []):
    for sent in data:
        for i, word in enumerate(sent):
            featureset.append((get_features(word, sent, i), word[3]))
    return featureset

def get_results(classifier, ref_results = collections.defaultdict(set), 
    test_results = collections.defaultdict(set)):
    for i, (features, label) in enumerate(training_featureset):
        ref_results[label].add(i)
        observed = classifier.classify(features)
        test_results[observed].add(i)
    return (ref_results, test_results)
        
training_featureset = get_featureset(training_set)
test_featureset = get_featureset(test_set)

bayes_classifier = nltk.NaiveBayesClassifier.train(training_featureset)
results = get_results(bayes_classifier)

print '''
Classifier accuracy (Bayes): %s
B Precision (Bayes): %s
B Recall (Bayes): %s
I Precision (Bayes): %s
I Recall (Bayes): %s
O Precision (Bayes): %s
O Recall (Bayes): %s
''' % (accuracy(bayes_classifier, test_featureset), 
       precision(results[0]['B-SNP'], results[1]['B-SNP']), 
       recall(results[0]['B-SNP'], results[1]['B-SNP']), 
       precision(results[0]['I-SNP'], results[1]['I-SNP']),
       recall(results[0]['I-SNP'], results[1]['I-SNP']),
       precision(results[0]['O'], results[1]['O']),
       recall(results[0]['O'], results[1]['O']))

#bayes_classifier.show_most_informative_features(10)

maxent_classifier = nltk.MaxentClassifier.train(training_featureset)
maxent_results = get_results(maxent_classifier)

print '''
Classifier accuracy (MaxEnt): %s
B Precision (MaxEnt): %s
B Recall (MaxEnt): %s
I Precision (MaxEnt): %s
I Recall (MaxEnt): %s
O Precision (MaxEnt): %s
O Recall (MaxEnt): %s
''' % (accuracy(maxent_classifier, test_featureset), 
       precision(maxent_results[0]['B-SNP'], maxent_results[1]['B-SNP']), 
       recall(maxent_results[0]['B-SNP'], maxent_results[1]['B-SNP']), 
       precision(maxent_results[0]['I-SNP'], maxent_results[1]['I-SNP']),
       recall(maxent_results[0]['I-SNP'], maxent_results[1]['I-SNP']),
       precision(maxent_results[0]['O'], maxent_results[1]['O']),
       recall(maxent_results[0]['O'], maxent_results[1]['O']))
       
#maxent_classifier.show_most_informative_features(10)
