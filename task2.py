import json
import os
import re
import csv
import nltk
import math
import copy
import pprint
import operator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# calculates the TFIDF for the ten claims matching the IDs listed below
# 75397, 150448, 214861, 156709, 83235, 129629, 149579, 229289, 33078, 6744
# It is important to note these do not match the spec's top 10 IDs for some reason

# then compares these tfidfs to those of the documents and pulls the top five likely
# documents for each claim based on the highest similarity score

#------------------------------------------------------------------------------#
#                            LOADING CLAIM DATA
#------------------------------------------------------------------------------#

# load the json paths

path_to_claims = 'data/train.jsonl'

# add each claim to an array (claims)

print("\nLoading claims...")

claims = []
claim_counter = 0

with open(path_to_claims,'r') as f:
    for line in f:
        claims.append(json.loads(line))
        
        #limits it to the first 10 claims which align with the 10 specified in the cw
        claim_counter += 1
        if claim_counter == 10:
            break

print("\nclaims loaded: ", len(claims), "/ 10\n") #should be 10

#------------------------------------------------------------------------------#
#                       LOAD DOCUMENTS FOR IDF PROCCESSING
#------------------------------------------------------------------------------#

# load the json paths

path_to_jsons = 'data/wiki-pages/wiki-pages'

json_files = [j for j in os.listdir(path_to_jsons) if j.endswith('.jsonl')] #gets list of all jsonl file paths from folder

# add each json to an array (data)

print("\nLoading documents...")

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
#            if countTEMP == 10000:
#                break
#    countTEMP2 += 1
#    if countTEMP2 == 3:
#        break
###################


print("Number of documents loaded: ", len(data), "\n")

#------------------------------------------------------------------------------#
#                       CALCULATING TF FOR CLAIMS
#------------------------------------------------------------------------------#

# creates a list of tokens from all claims' text field (as lowercase) after removing the stop words

print("\nProcessing claims...\n")

# claimsTF = {'ID': claimID, 'Terms': {'term': "", 'frequency': 0}}

claimsTF = {}
claimsIDF = {}
claimsTFIDF = {}
allWords = []
all_words_IDF = {}

for claim in claims:
    
    print("\n---", claim['id'], "---")
    print(claim['claim'])

    temp_text = re.sub(r'[^a-zA-Z\d\s]', "", str(claim['claim']))   # removes all non-alphanumeric characters

    print("Tokenizing claim...")
    
    words = word_tokenize(temp_text.lower()) # tokenizes the claim as lowercase

    print("Removing stop words...")

    stopWords = set(stopwords.words('english'))

    for stop_word in stopWords:
        stop_word = stop_word.lower()

    filtered_words = []

    for w in words:
        if w not in stopWords:
            filtered_words.append(w)

    print("Calculating TF...")

    wordsTF = dict.fromkeys(filtered_words, 0)

    # keeps track of all words for later calculating the documents tf/idf
    allWords.extend(wordsTF.keys())

    for word in filtered_words:
        wordsTF[word] += 1

    # handles strange occasional 0 count
    for word in wordsTF:
        if wordsTF[word] == 0:
            wordsTF[word] = 1

    # normalizes the TF based on this article: https://ishwortimilsina.com/calculate-tf-idf-vectors/
    for word in wordsTF:
        wordsTF[word] = 1 + math.log10(wordsTF[word])

#------------------------------------------------------------------------------#
#                       CALCULATING IDF FOR TERMS
#------------------------------------------------------------------------------#

    print("Searching for IDF...")

    wordsIDF = dict.fromkeys(wordsTF, 0)

    # for each word in this claim, searches through each document and counts the occurances of said word
    for word in wordsIDF:
        
        for entry in data:
            
            temp_text = re.sub(r'[^a-zA-Z\d\s]', "", entry['text'])
            
            if word in str(temp_text).lower():
                wordsIDF[word] += 1

    print("Calculating IDF...")

    # converts the count from prev step into the IDF using the formula: IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

    for word in wordsIDF:
        # prevents 0 division
        if wordsIDF[word] != 0:
            wordsIDF[word] = math.log10(float(len(data)) / (float(wordsIDF[word])))
            
        all_words_IDF.update({word:wordsIDF[word]})

#------------------------------------------------------------------------------#
#                CALCULATING (normalized) TFIDF FOR CLAIMS
#------------------------------------------------------------------------------#

    print("Calculating TFIDF...")

    wordsTFIDF = dict.fromkeys(wordsTF.keys())

    for word in wordsTF:
        wordsTFIDF[word] = wordsTF[word] * wordsIDF[word]

    print("Normalizing TFIDF...")

    normalization_denom = 0

    for word, tfidf in wordsTFIDF.items():
        normalization_denom += tfidf ** 2

    normalization_denom = math.sqrt(normalization_denom)

    for word, tfidf in wordsTFIDF.items():
        wordsTFIDF[word] = tfidf / normalization_denom

    print("Done.\n")

    claimsTF[claim['id']] = [wordsTF]
    claimsIDF[claim['id']] = [wordsIDF]
    claimsTFIDF[claim['id']] = [wordsTFIDF]

print("\nTF-IDF Calculated for all claims.\n")

#pprint.pprint(claimsTF)
#print("\n\n\n")
#pprint.pprint(claimsIDF)
#print("\n\n\n")
#pprint.pprint(claimsTFIDF)

#------------------------------------------------------------------------------#
#                       CALCULATING TF-IDF FOR DOCUMENTS
#------------------------------------------------------------------------------#

print("\nCalculating TF-IDF for documents...")
print("")

# removes potential duplicates in the claims' words list
allWords = list(dict.fromkeys(allWords))

documentTFIDF = {}
document_count = 0

for entry in data:
    
    document_wordTF = dict.fromkeys(allWords, 0)
    document_wordTFIDF = {}
    
    for word in allWords:
        
        text = re.sub(r'[^a-zA-Z\d\s]', "", entry['text'])
        
        # counts the inital TF based on each document's text
        document_wordTF[word] += text.lower().count(word)
        
        # normalizes the TF
        if document_wordTF[word] != 0:
            document_wordTF[word] = 1 + math.log10(document_wordTF[word])
        
        # calculates the TF-IDF using the IDF determined earlier
        document_wordTFIDF[word] = document_wordTF[word] * all_words_IDF[word]
    
    # normalization
    normalization_denom = 0

    for word, tfidf in document_wordTFIDF.items():
        normalization_denom += tfidf ** 2

    normalization_denom = math.sqrt(normalization_denom)

    for word, tfidf in document_wordTFIDF.items():
        if normalization_denom != 0:
            document_wordTFIDF[word] = tfidf / normalization_denom

    documentTFIDF[entry['id']] = [document_wordTFIDF]

print("\nTF-IDF Calculated for all documents.\n")

#------------------------------------------------------------------------------#
#                           COMPUTING SIMILARITY
#------------------------------------------------------------------------------#

print("\nCalculating similarity for claims and documents...")

# claims TFIDF is a key value pair where key is the claim ID and the value is another
# dictionary in the stlye of === key is word and value is TFIDF

final_claim_scores = [] # can't be dict because how I save = keys would repeat

# goes through each document by ID only
for documentID in documentTFIDF.keys():

    # goes through each claim to generate the sim score
    for claimID in claimsIDF.keys():
        simularity_score = 0

        # goes through each word for the sim score calculation
        # multiplies the word's claimtfidf by the documenttfidf
        # following: https://ishwortimilsina.com/calculate-cosine-similarities-queries-documents
        for word, word_tfidf in claimsTFIDF[claimID][0].items():
            simularity_score += documentTFIDF[documentID][0][word] * word_tfidf

        temp = (claimID, documentID, simularity_score)
        final_claim_scores.append(temp)

#------------------------------------------------------------------------------#
#                               SORTING TOP 5
#------------------------------------------------------------------------------#

# 75397, 150448, 214861, 156709, 83235, 129629, 149579, 229289, 33078, 6744
# its awful but it works

top_claim_scores = sorted(final_claim_scores, key=operator.itemgetter(0, 2), reverse=True)
top_results = []

# 75397 -----------------------------------------------------------------------#

results_75397 = []

for result in top_claim_scores:
    if result[0] == 75397:
        results_75397.append(result)

top_results_75397 = sorted(results_75397, key=operator.itemgetter(2), reverse=True)[:5]

print("\n\nClaim: 75397 || Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.")
pprint.pprint(top_results_75397)
print("\n")

for result in top_results_75397:
    top_results.append(result)

# 150448 -----------------------------------------------------------------------#

results_150448 = []

for result in top_claim_scores:
    if result[0] == 150448:
        results_150448.append(result)

top_results_150448 = sorted(results_150448, key=operator.itemgetter(2), reverse=True)[:5]

print("Claim: 150448 || Roman Atwood is a content creator.")
pprint.pprint(top_results_150448)
print("\n")

for result in top_results_150448:
    top_results.append(result)

# 214861 -----------------------------------------------------------------------#

results_214861 = []

for result in top_claim_scores:
    if result[0] == 214861:
        results_214861.append(result)

top_results_214861 = sorted(results_214861, key=operator.itemgetter(2), reverse=True)[:5]

print("Claim: 214861 || History of art includes architecture, dance, sculpture, music, painting, poetry literature, theatre, narrative, film, photography and graphic arts.")
pprint.pprint(top_results_214861)
print("\n")

for result in top_results_214861:
    top_results.append(result)

# 156709 -----------------------------------------------------------------------#

results_156709 = []

for result in top_claim_scores:
    if result[0] == 156709:
        results_156709.append(result)

top_results_156709 = sorted(results_156709, key=operator.itemgetter(2), reverse=True)[:5]

print("Claim: 156709 || Adrienne Bailon is an accountant.")
pprint.pprint(top_results_156709)
print("\n")

for result in top_results_156709:
    top_results.append(result)

# 129629 -----------------------------------------------------------------------#

results_129629 = []

for result in top_claim_scores:
    if result[0] == 129629:
        results_129629.append(result)

top_results_129629 = sorted(results_129629, key=operator.itemgetter(2), reverse=True)[:5]

print("Claim: 129629 || Homeland is an American television spy thriller based on the Israeli television series Prisoners of War.")
pprint.pprint(top_results_129629)
print("\n")

for result in top_results_129629:
    top_results.append(result)

# 149579 -----------------------------------------------------------------------#

results_149579 = []

for result in top_claim_scores:
    if result[0] == 149579:
        results_149579.append(result)

top_results_149579 = sorted(results_149579, key=operator.itemgetter(2), reverse=True)[:5]

print("Claim: 149579 || Beautiful reached number two on the Billboard Hot 100 in 2003.")
pprint.pprint(top_results_149579)
print("\n")

for result in top_results_149579:
    top_results.append(result)

# 6744 -----------------------------------------------------------------------#

results_6744 = []

for result in top_claim_scores:
    if result[0] == 6744:
        results_6744.append(result)

top_results_6744 = sorted(results_6744, key=operator.itemgetter(2), reverse=True)[:5]

print("Claim: 6744 || The Ten Commandments is an epic film.")
pprint.pprint(top_results_6744)
print("\n")

for result in top_results_6744:
    top_results.append(result)

# 33078 -----------------------------------------------------------------------#

results_33078 = []

for result in top_claim_scores:
    if result[0] == 33078:
        results_33078.append(result)

#pprint.pprint(sorted(results_33078, key=operator.itemgetter(2), reverse=True))

top_results_33078 = sorted(results_33078, key=operator.itemgetter(2), reverse=True)[:5]

print("Claim: 33078 || The Boston Celtics play their home games at TD Garden.")
pprint.pprint(top_results_33078)
print("\n")

for result in top_results_33078:
    top_results.append(result)

# 229289 -----------------------------------------------------------------------#

results_229289 = []

for result in top_claim_scores:
    if result[0] == 229289:
        results_229289.append(result)

top_results_229289 = sorted(results_229289, key=operator.itemgetter(2), reverse=True)[:5]

print("Claim: 229289 || Neal Schon was named in 1954")
pprint.pprint(top_results_229289)
print("\n")

for result in top_results_229289:
    top_results.append(result)

# 83235 -----------------------------------------------------------------------#

results_83235 = []

for result in top_claim_scores:
    if result[0] == 83235:
        results_83235.append(result)

top_results_83235 = sorted(results_83235, key=operator.itemgetter(2), reverse=True)[:5]

print("Claim: 83235 || System of a Down briefly disbanded in limbo.")
pprint.pprint(top_results_83235)
print("\n")

for result in top_results_83235:
    top_results.append(result)

#------------------------------------------------------------------------------#
#                               SAVING FILES
#------------------------------------------------------------------------------#

print("\nSaving claim TF to files...")

# as csv for submission
with open('task2_claim_TF.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key in claimsTF.keys():
        for value in claimsTF[key][0].items():
            writer.writerow([key, value[0], value[1]])

# as json for easy future loading
with open('task2_claim_TF_json.json', 'w') as json_file:
    json.dump(claimsTF, json_file)

print("Saving document TFIDF to files...")

# as csv for submission
with open('task2_document_TFIDF.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key in documentTFIDF.keys():
        for value in documentTFIDF[key][0].items():
            writer.writerow([key, value[0], value[1]])

# as json for easy future loading
with open('task2_document_TFIDF_json.json', 'w') as json_file:
    json.dump(documentTFIDF, json_file)

print("Files saved.\n")

print("\nSaving IDF to files...")

# as csv for submission
with open('task2_claim_IDF.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key in claimsIDF.keys():
        for value in claimsIDF[key][0].items():
            writer.writerow([key, value[0], value[1]])

# as json for easy future loading
with open('task2_claim_IDF_json.json', 'w') as json_file:
    json.dump(claimsIDF, json_file)

print("Files saved.\n")

print("\nSaving claim TFIDF to files...")

# as csv for submission
with open('task2_claim_TFIDF.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key in claimsTFIDF.keys():
        for value in claimsTFIDF[key][0].items():
            writer.writerow([key, value[0], value[1]])

# as json for easy future loading
with open('task2_claim_TFIDF_json.json', 'w') as json_file:
    json.dump(claimsTFIDF, json_file)

print("Files saved.\n")

print("\nSaving top five to files...")

# as csv for submission
with open('task2_top_documents.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for result in top_results:
        writer.writerow(result)

# as json for easy future loading
with open('task2_top_documents.json', 'w') as json_file:
    json.dump(top_results, json_file)

print("Files saved.\n")
