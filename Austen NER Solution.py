# Preparation
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from collections import Counter

modules = ["averaged_perceptron_tagger", "maxent_ne_chunker", "punkt"]

for module in modules:
    nltk.download(module)

    
# Read in text
with open('Austen - Pride and Prejudice.txt') as file_in:
    austen_text = file_in.read()


# Named Entity Recognition
austen_sents = sent_tokenize(austen_text)
austen_words = [word_tokenize(sent) for sent in austen_sents]
austen_pos = [pos_tag(sent) for sent in austen_words]
austen_ner = [ne_chunk(sent) for sent in austen_pos]
austen_chunks = [chunk for sent in austen_ner for chunk in sent]
austen_persons = [chunk.leaves() for chunk in austen_chunks if type(chunk)==nltk.tree.Tree and chunk.label()=='PERSON']
austen_names_only = [name for person in austen_persons for name,tag in person]
austen_counted = Counter(austen_names_only)
print(austen_counted.most_common())
