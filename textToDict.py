import os

eng_index = 0
ger_index = 0

#dictionaries with all key value pairs(contains duplicate values, unique keys)
english_dict = {}
german_dict = {}

#final dictionaries without duplicates
eng_dict_clean = {}
ger_dict_clean = {}

def _create_dictionary(src_foldername):
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


##use string.lower() to turn a given string into lower case characters##

###dict.update(dict2)
###Adds dictionary dict2's key-values pairs to dict
