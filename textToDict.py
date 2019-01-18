import os
import numpy as np

def _create_dictionary(src_foldername):
    
    eng_index = 0
    ger_index = 0

    #dictionaries with all key value pairs(contains duplicate values, unique keys)
    english_dict = {}
    german_dict = {}

    #final dictionaries without duplicates
    eng_dict_clean = {}
    ger_dict_clean = {}

    #special tags for encoding and decoding in the nmt need to be added
    eng_dict_clean[eng_index] = '<s>'
    eng_index = eng_index +1
    eng_dict_clean[eng_index] = '</s>'
    eng_index = eng_index +1
    eng_dict_clean[eng_index] = '<space>'
    eng_index = eng_index +1
    
    ger_dict_clean[ger_index] = '<s>'
    ger_index = ger_index + 1
    ger_dict_clean[ger_index] = '</s>'
    ger_index = ger_index + 1
    ger_dict_clean[ger_index] = '<space>'
    ger_index = ger_index + 1
    
    
    for filename in os.listdir(src_foldername):
        file = open(src_foldername + '/' + filename)
        #con file structure:
        #annot_eng	SATURDAY EVENING CINEMA X-right START WHEN?
        #annot_deu	SAMSTAG ABEND KINO X-rechts ANFANGEN WANN?
        #transl_eng	When does the film start on Saturday night?
        #transl_deu	Wann fängt am Samstag Abend das Kino an?

        #iso file structure
        #annot_eng	START
        #annot_deu	ANFANGEN

        if 'con' in filename:
            #operations to split the strings read from the file and clean them up
            lines = file.readlines()
            eng1 = lines[2]
            ger1 = lines[3]
            engwords = eng1.split()
            gerwords = ger1.split()
            engwords.remove('transl_eng')
            gerwords.remove('transl_deu')
        
            #remove punctuation
            engwords[len(engwords)-1] = engwords[len(engwords)-1][0:engwords[len(engwords)-1].count('')-2]
            gerwords[len(gerwords)-1] = gerwords[len(gerwords)-1][0:gerwords[len(gerwords)-1].count('')-2]

        if 'iso' in filename:
            lines = file.readlines()
            eng1 = lines[0]
            ger1 = lines[1]
            
            engwords = eng1.split()
            engwords = engwords[1].split('|')
            gerwords = ger1.split()
            gerwords = gerwords[1].split('|')
            
        for word in engwords:
            english_dict[eng_index] = word.lower()
            eng_index = eng_index +1

        for word in gerwords:
            german_dict[ger_index] = word.lower()
            ger_index = ger_index +1


    #remove duplicate values
    for key,value in english_dict.items():
        if value not in eng_dict_clean.values():
            eng_dict_clean[key] = value
            
    for key,value in german_dict.items():
        if value not in ger_dict_clean.values():
            ger_dict_clean[key] = value

    #create inverse dictionaries
    english_inv_dict = {value:key for key,value in eng_dict_clean.items()}
    german_inv_dict = {value:key for key,value in ger_dict_clean.items()}

    return eng_dict_clean, english_inv_dict, ger_dict_clean, german_inv_dict



def _embed_sentence_data(src_foldername, max_sentence_length, eng_inv_dict, ger_inv_dict):
    #Returns sentences translated to indices as numpy array

    
    eng_sentence_list = []
    ger_sentence_list = []
    eng_indices = []
    ger_indices = []
    
    for filename in os.listdir(src_foldername):

        #reset the sentence containers
        eng_indices = []
        ger_indices = []

        if 'con' in filename:
            
            file = open(src_foldername + '/' + filename)
            #operations to split the strings read from the file and clean them up
            lines = file.readlines()
            eng1 = lines[2]
            ger1 = lines[3]
            engwords = eng1.split()
            gerwords = ger1.split()
            engwords.remove('transl_eng')
            gerwords.remove('transl_deu')
          
            #remove punctuation
            engwords[len(engwords)-1] = engwords[len(engwords)-1][0:engwords[len(engwords)-1].count('')-2]
            gerwords[len(gerwords)-1] = gerwords[len(gerwords)-1][0:gerwords[len(gerwords)-1].count('')-2]

            #Turn words into indices
            for word in engwords:
                eng_indices.append(eng_inv_dict[word.lower()])

            eng_indices.append(eng_inv_dict['</s>'])
            ger_indices.append(ger_inv_dict['<s>'])

            for word in gerwords:
                ger_indices.append(ger_inv_dict[word.lower()])

            #add the index for padding to the list
            while len(eng_indices) < max_sentence_length :
                eng_indices.append(eng_inv_dict['<space>'])
            
            while len(ger_indices) < max_sentence_length :
                ger_indices.append(ger_inv_dict['<space>'])
                
            #bundle all sentences to a matrix(numpy array)
            eng_sentence_list.append(eng_indices)
            ger_sentence_list.append(ger_indices)

    return np.asarray(eng_sentence_list), np.asarray(ger_sentence_list)


        
