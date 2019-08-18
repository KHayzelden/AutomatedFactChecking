import json
import os
import re
import csv
import nltk
import math
import pprint
import operator
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# calcualtes the tf

# tf is the number of times the term appears in a document divided by the total number of terms in the document (for normalization)
# TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).


#------------------------------------------------------------------------------#
#                               LOADING DATA
#------------------------------------------------------------------------------#

# load the json paths

path_to_jsons = 'data/wiki-pages/wiki-pages'

json_files = [j for j in os.listdir(path_to_jsons) if j.endswith('.jsonl')] #gets list of all jsonl file paths from folder

# add each json to an array (data)

print("Loading documents...")

data = []
countTEMP = 0
countTEMP2 = 0

for path in json_files:
    newPath = 'data/wiki-pages/wiki-pages/' + path
    with open(newPath,'r') as f:
        for line in f:
            data.append(json.loads(line))
            
################### temp only some jsonls
#            countTEMP += 1
#            if countTEMP == 100000:
#                break
#    countTEMP2 += 1
#    if countTEMP2 == 1:
#        break
###################

print("Number of documents loaded: ", len(data))

#------------------------------------------------------------------------------#
#                               CALCULATING TF
#------------------------------------------------------------------------------#

# creates a list of tokens from all entries' text field (as lowercase to remove issues with "the" vs "The")

print("Loading text...")

text = ""
counter = 0

for entry in data:
    temp_text = re.sub(r'[^a-zA-Z\d\s]', "", str(entry['text']))   # removes all non-alphanumeric characters
    text += temp_text.lower()
    counter += 1

print("Loaded texts: ", counter)

print("Tokenizing...")

words = word_tokenize(text) #gives a list of every word in the corpus without removing duplicates

print("Number of words counted: ", len(words))

# counts the occurances of each word in words

print("Calculating unique words...")

wordsTF = dict.fromkeys(words, 0)

for w in words:
    wordsTF[w] += 1

print("Number of unique words counted: ", len(wordsTF))

# graphs the tf as zipfs law

term_frequencies = list(wordsTF.items())

sorted_term_frequencies = sorted(term_frequencies, key=operator.itemgetter(1), reverse=True)

x = []
y = []
labels = []
x_axis = []
counter = 1

# for plotting zipf's law, x axis is log(rank order) and y is log(frequency)
for term in sorted_term_frequencies:
    x.append(math.log10(counter))
    x_axis.append(counter)
    y.append(math.log10(term[1]))
    labels.append(term[0])
    counter += 1

#plt.scatter(x,y)

fig, plot = plt.subplots()
plot.scatter(x, y, marker='.' , color='red')

counter = 0

for i, txt in enumerate(labels):
    plot.annotate(txt, (x[i], y[i]))
    counter += 1
    if counter == 5:
        break

plot.axes.xaxis.set_ticklabels([])

plt.xlabel('Term Rank  (Highest Rank to Lowest Rank)')
plt.ylabel('Term Frequency')
plt.title('Zipf\'s Law')
plt.grid(True)
plt.show()


# calculates the normalized tf for the word based on its number of occurances vs total words in text (len(words))
print("Saving to file...")

with open('task1_raw_TF.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in wordsTF.items():
        writer.writerow([key, value])

print("File saved.")

print("Calculating TF...")

for w in wordsTF:
    wordsTF[w] = wordsTF[w] / float(len(words))

print("TF calculations completed.")

# writes TF results to file

print("Saving to file...")

with open('task1_normalized_TF.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in wordsTF.items():
        writer.writerow([key, value])

print("File saved.")
