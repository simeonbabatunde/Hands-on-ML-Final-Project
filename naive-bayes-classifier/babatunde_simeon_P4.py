#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:00:19 2019

@author: Simeon Babatunde

Description: 
    This script contains the implementation of a Naive Bayes classification algorithm for classifying email subject line 
    as a SPAM or HAM. The program prompts user for the name of a training set file and the name of the file containing Stop Words.
    The program then creates a vocabulary of words found in the subject lines of the training set associated with an estimated probability 
    of each word appearing in a Spam and Ham email. The program then prompts the user for a labeled test set and predict the class 
    (1 = Spam, 0 = Ham) of each subject line using a Naïve Bayes approach.
"""   
import numpy as np 
import pandas as pd

def clean_text(text):
    '''
    This function takes in a line of text and removes unnecessary characters 
    Args: string containing a line of texts
    Returns: a cleaned version of the text input
    '''
    text = text.lower().strip()
    for char in text:
        if char in """[].,"!@;:#$%^&*()+/?{}|""":
            text = text.replace(char, " ")
        if char in """—'-_\x92x93x94""":
            text = text.replace(char, "")
            
    return text


def spamham_word_counter(text, word_class, word_dictionary):
    '''
    This function counts how many time the each word in a text appears in ham or spam  
    Args: 
        text: text containing the words to be counted
        word_class: This could be 0 (ham) or 1 (spam)
        word_dictionary: A dict to keep the unique words and how often they appear in ham or spam
    Returns: 
        A dict containing counted words
    '''
    for each_word in text:
        if each_word in word_dictionary:
            if word_class == 1:
                word_dictionary[each_word][1] += 1
            else:
                word_dictionary[each_word][0] += 1
        else:
            if word_class == 1:
               word_dictionary[each_word] = [0, 1] 
            else:
               word_dictionary[each_word] = [1, 0] 
            
    return word_dictionary


def pseudocount_smoothning(k, wordscounted, nham, nspam):
    '''
    This function performs data smoothening to avoid zero probability problem  
    Args: 
        k: Pseudocount used for smothening
        wordscounted: Dictionary of the counted words that needs smoothening 
        nham: number of hams
        nspam: number of spams
    Returns: 
        A smoothed version of the wordscounted dictionary
    '''
    for each_key in wordscounted:
        wordscounted[each_key][0] = (wordscounted[each_key][0] + k)/(2*k + nham)
        wordscounted[each_key][1] = (wordscounted[each_key][1] + k)/(2*k + nspam)
        
    return wordscounted
    

def compute_correctness(prediction_array, test_labels):
    '''
    This function computes the number of correct predictions made by the algorithm; between predictions and test labels
    Args: 
        prediction_array: A numpy array of predicted class
        test_labels: A numpy array of labels 
    Returns: 
        Number of correct predictions made
    '''
    correct = np.count_nonzero(prediction_array == test_labels)
    return correct



################## Main program Starts Here ###################################
if __name__ == "__main__":
    spam, ham = 0, 0
    test_spam, test_ham = 0, 0
    counted_words = dict()
    labels = []
    predictions = []
    
    # Prompt user for filename containing training dataset and read it in
    train_file = input("Enter the name of the file containing the training data: ")
    stopwords_file = input("Enter the name of the file containing the stop words: ")
    
    if (not train_file) or (not stopwords_file):
        print("File names cannot be empty, please try again!")
        exit(0)
    
    try:
        with open(train_file, "r", encoding = 'unicode-escape') as tfileobj, open(stopwords_file, "r") as swfileobj:
            stop_words = [line.strip() for line in swfileobj.readlines()] 
            stop_words = [ln for ln in stop_words if ln]
            
            text_line = tfileobj.readline()
            while text_line != "":
                is_spam = int(text_line[:1])
                
                if is_spam == 1:
                    spam += 1
                else:
                    ham += 1
                cleaned_text = clean_text(text_line[1:])
                words = cleaned_text.split()
                words = set(words)          # Remove duplocate words using set
                words = words.difference(stop_words)  # Remove stop words
                counted_words = spamham_word_counter(words, is_spam, counted_words)
                text_line = tfileobj.readline()
        
    except:
        print("Cannot read one or both files. Ensure {} and {} are located in this directory".format(train_file,stopwords_file))
        exit(0)
    

    # Create word Vocabulary
    vocabulary = pseudocount_smoothning(1, counted_words, ham, spam)
    
    # Compute Naive Bayes Paramenters
    prob_not_spam = float(ham)/(spam + ham)
    prob_spam = float(spam)/(spam + ham)
    
    # Read in the test data
    test_file = input("Enter the name of the file containing test data: ")
    try:
        with open(test_file, "r", encoding = "unicode-escape") as tstfileobj:
             txt_line = tstfileobj.readline()
             while txt_line != "":
                 is_test_spam = int(txt_line[:1])
                 labels.append(is_test_spam)
                 
                 if is_test_spam == 1:
                     test_spam += 1
                 else:
                     test_ham += 1
                 
                 clean_test_data = clean_text(txt_line[1:])
                 test_subject_line = clean_test_data.split()
                 test_subject_line = set(test_subject_line)
                 test_subject_line = test_subject_line.difference(stop_words)
                 
                 # Compute remaining Naive Bayes parameters
                 prob_subject_in_spam = 0.0
                 prob_subject_not_in_spam = 0.0
                 
                 for vocab in vocabulary:
                     if vocab in test_subject_line:
                         prob_subject_in_spam += np.log(vocabulary[vocab][1]) 
                         prob_subject_not_in_spam += np.log(vocabulary[vocab][0]) 
                     else:
                         prob_subject_in_spam += np.log(1 - vocabulary[vocab][1])  
                         prob_subject_not_in_spam += np.log(1 - vocabulary[vocab][0])
                         
                 prob_subject_in_spam = np.exp(prob_subject_in_spam)
                 prob_subject_not_in_spam = np.exp(prob_subject_not_in_spam)
                 
                 # Compute Naive Bayes probability
                 bayes_prob_spam = (prob_subject_in_spam * prob_spam)/((prob_subject_in_spam * prob_spam) + (prob_subject_not_in_spam * prob_not_spam))
                 predictions.append(1) if bayes_prob_spam >= 0.5 else predictions.append(0)
                 
                 txt_line = tstfileobj.readline()
                 
             print("\nSpam emails: {} \nHam emails: {}\n".format(test_ham, test_spam))
    except:
        print("Cannot read test file. Ensure {} is located in this directory".format(test_file))
        exit(0)
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    correct_results = compute_correctness(predictions, labels)
    accuracy = float(correct_results)/(test_ham + test_spam)
    
    # Compute Confusion Matrix
    y_actu = pd.Series(labels, name='Actual')
    y_pred = pd.Series(predictions, name='Predicted')
    confusion_matrix = pd.crosstab(y_actu, y_pred)
    true_negative = confusion_matrix[0][0]
    false_negative = confusion_matrix[0][1]
    false_positive = confusion_matrix[1][0]
    true_positive = confusion_matrix[1][1]
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f1 = 2 * ((precision * recall)/(precision + recall))
    
    # Print results to the screen
    print("False Positives: {}".format(false_positive))
    print("True Psotives: {}".format(true_positive))
    print("False Negatives: {}".format(false_negative))
    print("True Negatives: {}\n".format(true_negative))
    
    print("Accuracy: {:0.2f}%".format(100 * accuracy))
    print("Precision: {:0.2f}".format(precision))
    print("Recall: {:0.2f}".format(recall))
    print("F1 Score: {:0.2f}".format(f1))