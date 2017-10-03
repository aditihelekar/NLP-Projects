# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:01:14 2017

@author: aditi
"""

from nltk.corpus import udhr
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist, ConditionalFreqDist
import string
import csv

def get_Unigrams(data):
    unigrams_list = []
    for word in data:
        for character in word:
            #print(character)
            unigrams_list.append(character)
    return unigrams_list

def get_Bigrams(data):
    bigrams_list = []
    for word in data:
        for i in range(len(word) - 1):
            #print(word[i], word[i+1])
            bigrams_list.append((word[i] , word[i+1]))
    return bigrams_list


def get_Trigrams(data):
    trigrams_list = []
    for word in data:
        for i in range(len(word) - 2):
            trigrams_list.append((word[i] , word[i+1], word[i+2]))
    return trigrams_list

#Function clean text
def cleanText(Sentence):
    #Remove punctuation
    punc = Sentence.maketrans({key: None for key in string.punctuation})
    new_sentence = Sentence.translate(punc)
    #Remove Capitalization
    new_sentence = new_sentence.lower()
    #Tokenize words
    tokenizedWords = word_tokenize(new_sentence)
    
    #text = "_".join(tokenizedWords)
    #text = text.replace(" ", "_")
    
    #print(text)
    return (" ".join(tokenizedWords))

def calculateUnigramProbability(word, language_type):
    unigramDict = prob_eng_uni
    probability = 1
    if language_type == 0:
        unigramDict = prob_eng_uni
    elif language_type == 1:
        unigramDict = prob_fre_uni
    elif language_type == 2:
        unigramDict = prob_ita_uni
    elif language_type == 3:
        unigramDict = prob_spa_uni
    #print(word)
    for i in word:
        #print(i)
        if unigramDict.get(i) != None:
            probability = probability * unigramDict.get(i)
        else:
            probability = probability * 0
    #print(probability)    
    return probability

def probabilityUnigram(unigrams, language_type):
    fdist_uni = FreqDist(unigrams)
    count = 0
    if language_type == 0:
        count = uni_eng_count
    elif language_type == 1:
        count = uni_fre_count
    elif language_type == 2:
        count = uni_ita_count
    elif language_type == 3:
        count = uni_spa_count
        
    word_dict = {}
    for word,freq in fdist_uni.items():
        #print(word,freq)
        word_dict[word] = freq/count
    print(word_dict)
    return word_dict

def probabilityBigram(bigrams,language_type):
    fdist_bi =FreqDist(bigrams) 
    unigramFreq = fdist_eng_uni
    word_dict = {}
    if language_type == 0:
        unigramFreq = fdist_eng_uni
    elif language_type == 1:
        unigramFreq = fdist_fre_uni
    elif language_type == 2:
        unigramFreq = fdist_ita_uni
    elif language_type == 3:
        unigramFreq = fdist_spa_uni
        
    for word,freq in fdist_bi.items():
        #print(word)
        word_dict[word] = freq/unigramFreq[word[0]]
        
    #print("Bigram Dictionary is: ")
    print(word_dict)
    return word_dict

def probabilityTrigram(trigrams,language_type):
    fdist_tri = FreqDist(trigrams) 
    bigramFreq = fdist_eng_bi
    word_dict = {}
    if language_type == 0:
        bigramFreq = fdist_eng_bi
    elif language_type == 1:
        bigramFreq = fdist_fre_bi
    elif language_type == 2:
        bigramFreq = fdist_ita_bi
    elif language_type == 3:
        bigramFreq = fdist_spa_bi
        
    for word,freq in fdist_tri.items():
        #print(word)
        temp = (word[0],word[1])
        #print(temp)
        for dictW,freq1 in bigramFreq.items():
            if temp == dictW:
                #print(dictW, freq1)
                word_dict[word] = freq/freq1
    print(word_dict)
    return word_dict

english = udhr.raw('English-Latin1') 
french = udhr.raw('French_Francais-Latin1') 
italian = udhr.raw('Italian_Italiano-Latin1') 
spanish = udhr.raw('Spanish_Espanol-Latin1') 

english = cleanText(english)
french = cleanText(french)
italian = cleanText(italian)
spanish = cleanText(spanish)


# Train and Development sets
english_train, english_dev = english.split()[0:1000], english.split()[1000:1100]
french_train, french_dev = french.split()[0:1000], french.split()[1000:1100]
italian_train, italian_dev = italian.split()[0:1000], italian.split()[1000:1100]
spanish_train, spanish_dev = spanish.split()[0:1000], spanish.split()[1000:1100]

#english_train = (" ".join(english_train)).replace(" ","_")
english_dev = " ".join(english_dev)
#french_train = (" ".join(french_train)).replace(" ","_")
french_dev = " ".join(french_dev)
#italian_train = (" ".join(italian_train)).replace(" ","_")
italian_dev = " ".join(italian_dev)
#spanish_train = (" ".join(spanish_train)).replace(" ","_")
spanish_dev = " ".join(spanish_dev)

# Test sets
english_test_set = udhr.words('English-Latin1')[0:1000]
#english_test = cleanText(english_test_set).split()[0:1000]

french_test_set = udhr.words('French_Francais-Latin1')[0:1000]
#french_test = cleanText(french_test_set).split()[0:1000]

italian_test_set = udhr.words('Italian_Italiano-Latin1')[0:1000]
#italian_test = cleanText(italian_test_set).split()[0:1000]

spanish_test_set = udhr.words('Spanish_Espanol-Latin1')[0:1000]
#spanish_test = cleanText(spanish_test_set).split()[0:1000]

########################  Unigrams   ###########################
#English Unigrams
print("English Unigram Model:")
eng_unigrams = get_Unigrams(english_train)
uni_eng_count = len(eng_unigrams)
fdist_eng_uni = FreqDist(eng_unigrams)
prob_eng_uni = probabilityUnigram(eng_unigrams, 0)
#Frenech unigrams
print("\nFrench Unigram Model:")
fre_unigrams = get_Unigrams(french_train)
uni_fre_count = len(fre_unigrams)
fdist_fre_uni = FreqDist(fre_unigrams)
prob_fre_uni = probabilityUnigram(fre_unigrams, 1)
#Italian unigrams
print("\nItalian Unigram Model:")
ita_unigrams = get_Unigrams(italian_train)
uni_ita_count = len(ita_unigrams)
fdist_ita_uni = FreqDist(ita_unigrams)
prob_ita_uni = probabilityUnigram(ita_unigrams, 2)
#Spanish inigrams
print("\nSpanish Unigram Model:")
spa_unigrams = get_Unigrams(spanish_train)
uni_spa_count = len(spa_unigrams)
fdist_spa_uni = FreqDist(spa_unigrams)
prob_spa_uni = probabilityUnigram(spa_unigrams, 3)

####################  Bigrams   ################################
#English Bigrams
print("\nEnglish Bigram Model:")
eng_bigrams = get_Bigrams(english_train)
prob_eng_bi = probabilityBigram(eng_bigrams,0)
fdist_eng_bi = FreqDist(eng_bigrams)
#Frenech Bigrams
print("\nFrench Bigram Model:")
fre_bigrams = get_Bigrams(french_train)
prob_fre_bi = probabilityBigram(fre_bigrams,1)
fdist_fre_bi = FreqDist(fre_bigrams)
#Italian Bigrams
print("\nItalian Bigram Model:")
ita_bigrams = get_Bigrams(italian_train)
fdist_ita_bi = FreqDist(ita_bigrams)
prob_ita_bi = probabilityBigram(ita_bigrams,2)
#Spanish Bigrams
print("\nSpanish Bigram Model:")
spa_bigrams = get_Bigrams(spanish_train)
fdist_spa_bi = FreqDist(spa_bigrams)
prob_spa_bi = probabilityBigram(spa_bigrams,3)


########################  Trigrams   ###########################
#English Trigrams
print("\nEnglish Trigram Model:")
eng_trigrams = get_Trigrams(english_train)
fdit_eng_tri = FreqDist(eng_trigrams)
prob_eng_tri = probabilityTrigram(eng_trigrams, 0)
#French Trigrams
print("\nFrench Trigram Model:")
fre_trigrams = get_Trigrams(french_train)
fdit_fre_tri = FreqDist(fre_trigrams)
prob_fre_tri = probabilityTrigram(fre_trigrams, 1)
#Italian Trigrams
print("\nItalian Trigram Model:")
ita_trigrams = get_Trigrams(italian_train)
fdit_ita_tri = FreqDist(ita_trigrams)
prob_ita_tri = probabilityTrigram(ita_trigrams, 2)
#Spanish Trigrams
print("\nSpanish Trigram Model:")
spa_trigrams = get_Trigrams(spanish_train)
fdit_spa_tri = FreqDist(spa_trigrams)
prob_spa_tri = probabilityTrigram(spa_trigrams, 3)


##################################################################################################

# Calculate probability on Test set

def getProbabilityUnigram(testSet, unigramModel1, unigramModel2, predictionLanguage, secondLanguage):
    
    wordList = []
    correctPredictionCount = 0
    punctuationMarks = [",", ".", "'", "!"]
    for word in testSet:
        #print(word)
        if word not in punctuationMarks:
            #print(word)
            for i in word:
                #print(i)
                probability_1 = 1
                probability_2 = 1
                # Calculate probability for first model
                if unigramModel1.get(i) != None:
                    probability_1 = probability_1 * unigramModel1.get(i)
                else:
                    probability_1 = probability_1 * 0
                
                # Calculate probability for second model
                if unigramModel2.get(i) != None:
                    
                    probability_2 = probability_2 * unigramModel2.get(i)
                else:
                    probability_2 = probability_2 * 0
                   
            #print(probability_1, probability_2)
            if probability_1 > probability_2:
                wordTuple = (word, probability_1, probability_2, predictionLanguage)
                correctPredictionCount = correctPredictionCount + 1
                wordList.append(wordTuple)
            else:
                wordTuple = (word, probability_1, probability_2, secondLanguage)
                wordList.append(wordTuple)
            
            accuracy = (correctPredictionCount / 1000) 
        
               
    return [wordList, accuracy]

print("------------------------ Unigram Accuracies ---------------------------")

englishUniTest = getProbabilityUnigram(english_test_set, prob_eng_uni, prob_fre_uni, "English", "French")
print("Accuracy of English Unigram model is " , englishUniTest[1] * 100, "%")
with open('englishUnigram.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Word','Probability 1', 'Probability 2', 'Predicted Language'])
    for row in englishUniTest[0]:
        csv_out.writerow(row)

frenchUniTest = getProbabilityUnigram(french_test_set, prob_fre_uni, prob_eng_uni, "French", "English")
print("Accuracy of French Unigram model is " , frenchUniTest[1] * 100, "%")
      
with open('frenchUnigram.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Word','Probability 1', 'Probability 2', 'Predicted Language'])
    for row in frenchUniTest[0]:
        csv_out.writerow(row)
        
italianUniTest = getProbabilityUnigram(italian_test_set, prob_ita_uni, prob_spa_uni, "Italian", "Spanish")
print("Accuracy of Italian Unigram model is " , italianUniTest[1] * 100, "%")
with open('italianUnigram.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Word','Probability 1', 'Probability 2', 'Predicted Language'])
    for row in italianUniTest[0]:
        csv_out.writerow(row)

spanishUniTest = getProbabilityUnigram(spanish_test_set, prob_spa_uni, prob_ita_uni, "Spanish", "Italian")
print("Accuracy of Spanish Unigram model is " , spanishUniTest[1] * 100, "%")
        
with open('spanishUnigram.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Word','Probability 1', 'Probability 2', 'Predicted Language'])
    for row in spanishUniTest[0]:
        csv_out.writerow(row)

        
        
def getProbabilityBigram(testSet, bigramModel1, bigramModel2, predictionLanguage, secondLanguage):
    
    wordList = []
    correctPredictionCount = 0
    punctuationMarks = [",", ".", "'", "!"]
    for word in testSet:
        #print(word)
        if word not in punctuationMarks:
            for i in range(len(word) - 1):
                #print(i)
                probability_1 = 1
                probability_2 = 1
                # Calculate probability for first model
                #print(word[i])
                #print(word[i], word[i+1], bigramModel1.get((word[i], word[i+1])))
                if bigramModel1.get((word[i], word[i+1])) != None:
                    probability_1 = probability_1 * bigramModel1.get((word[i], word[i+1]))
                else:
                    probability_1 = probability_1 * 0
                #print(word[i], bigramModel1.get((word[i], word[i+1])))
                # Calculate probability for second model
                #print(word[i], word[i+1], bigramModel2.get((word[i], word[i+1])))
                if bigramModel2.get((word[i], word[i+1])) != None:
                    
                    probability_2 = probability_2 * bigramModel2.get((word[i], word[i+1]))
                else:
                    probability_2 = probability_2 * 0
                    
            #print(probability_1, probability_2)
            if probability_1 > probability_2:
                wordTuple = (word, probability_1, probability_2, predictionLanguage)
                correctPredictionCount = correctPredictionCount + 1
                wordList.append(wordTuple)
            else:
                wordTuple = (word, probability_1, probability_2, secondLanguage)
                wordList.append(wordTuple)
            
            accuracy = (correctPredictionCount / 1000) 
            
        
               
    return [wordList, accuracy]


print("------------------------- Bigram Accuracies ---------------------------")
englishBiTest = getProbabilityBigram(english_test_set, prob_eng_bi, prob_fre_bi, "English", "French")
print("Accuracy of English Bigram model is " , englishBiTest[1] * 100, "%")
with open('englishBigram.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Word','Probability 1', 'Probability 2', 'Predicted Language'])
    for row in englishBiTest[0]:
        csv_out.writerow(row)

frenchBiTest = getProbabilityBigram(french_test_set, prob_fre_bi, prob_eng_bi, "French", "English")
print("Accuracy of French Bigram model is " , frenchBiTest[1] * 100, "%")
with open('frenchBigram.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Word','Probability 1', 'Probability 2', 'Predicted Language'])
    for row in frenchBiTest[0]:
        csv_out.writerow(row)


italianBiTest = getProbabilityBigram(italian_test_set, prob_ita_bi, prob_spa_bi, "Italian", "Spanish")
print("Accuracy of Italian Bigram model is " , italianBiTest[1] * 100, "%")
with open('italianBigram.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Word','Probability 1', 'Probability 2', 'Predicted Language'])
    for row in italianBiTest[0]:
        csv_out.writerow(row)

spanishBiTest = getProbabilityBigram(spanish_test_set, prob_spa_bi, prob_ita_bi, "Spanish", "Italian")
print("Accuracy of Spanish Bigram model is " , spanishBiTest[1] * 100, "%")
with open('spanishBigram.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Word','Probability 1', 'Probability 2', 'Predicted Language'])
    for row in spanishBiTest[0]:
        csv_out.writerow(row)
        
def getProbabilityTrigram(testSet, trigramModel1, trigramModel2, predictionLanguage, secondLanguage):
    
    wordList = []
    correctPredictionCount = 0
    punctuationMarks = [",", ".", "'", "!"]
    for word in testSet:
        #print(word)
        if word not in punctuationMarks:
            for i in range(len(word) - 2):
                #print(i)
                probability_1 = 1
                probability_2 = 1
                # Calculate probability for first model
                #print(word[i])
                #print(trigramModel1.get((word[i], word[i+1], word[i+2])))
                if trigramModel1.get((word[i], word[i+1], word[i+2])) != None:
                    probability_1 = probability_1 * trigramModel1.get((word[i], word[i+1], word[i+2]))
                else:
                    probability_1 = probability_1 * 0
                
                # Calculate probability for second model
                if trigramModel2.get((word[i], word[i+1], word[i+2])) != None:
                    
                    probability_2 = probability_2 * trigramModel2.get((word[i], word[i+1], word[i+2]))
                else:
                    probability_2 = probability_2 * 0
                    
            #print(probability_1, probability_2)
            if probability_1 > probability_2:
                wordTuple = (word, probability_1, probability_2, predictionLanguage)
                correctPredictionCount = correctPredictionCount + 1
                wordList.append(wordTuple)
            else:
                wordTuple = (word, probability_1, probability_2, secondLanguage)
                wordList.append(wordTuple)
            
            accuracy = (correctPredictionCount / 1000) 
               
    return [wordList, accuracy]

print("---------------------- Trigram Accuracies --------------------------")
englishTriTest = getProbabilityTrigram(english_test_set, prob_eng_tri, prob_fre_tri, "English", "French")
print("Accuracy of English Trigram model is " , englishTriTest[1] * 100, "%")
with open('englishTrigram.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Word','Probability 1', 'Probability 2', 'Predicted Language'])
    for row in englishTriTest[0]:
        csv_out.writerow(row)

frenchTriTest = getProbabilityTrigram(french_test_set, prob_fre_tri, prob_eng_tri, "French", "English")
print("Accuracy of French Trigram model is " , frenchTriTest[1] * 100, "%")
with open('fenchTrigram.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Word','Probability 1', 'Probability 2', 'Predicted Language'])
    for row in frenchTriTest[0]:
        csv_out.writerow(row)

italianTriTest = getProbabilityTrigram(italian_test_set, prob_ita_tri, prob_spa_tri, "Italian", "Spanish")
print("Accuracy of Italian Trigram model is " , italianTriTest[1] * 100, "%")
with open('italianTrigram.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Word','Probability 1', 'Probability 2', 'Predicted Language'])
    for row in italianTriTest[0]:
        csv_out.writerow(row)

spanishTriTest = getProbabilityTrigram(spanish_test_set, prob_spa_tri, prob_ita_tri, "Spanish", "Italian")
print("Accuracy of Spanish Trigram model is " , spanishTriTest[1] * 100, "%")
with open('spanishTrigram.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['Word','Probability 1', 'Probability 2', 'Predicted Language'])
    for row in spanishTriTest[0]:
        csv_out.writerow(row)