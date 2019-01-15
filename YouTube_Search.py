# Please check in this code
# python search.py / --q = surfing / --max-results = 10 / totalResults

import argparse
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
import pprint
import json
import pandas as pd
#from PP import S1, S2
sentence1 = ""
sentence2 = ""
for i in range(0,9):
    print('For pair:',i)

    sentence1 = ['I like that bachelor','I have a pen','John is very nice','It is a dog','It is a dog','It is a dog','Canis familiaris are animals','I have a hammer','I have a hammer'][i]
    sentence2 = ['I like that unmarried man','Where do you live?','Is John very nice?','It is a log','It is a pig','That must be your dog','Dogs are common pets','Take some nails','Take some apples'][i]
    print('sentence1')
    print(sentence1)
    print('sentence2')
    print(sentence2)



# NOTE: To use the sample, you must provide a developer key obtained in the Google APIs Console. 
# Search for DEVELOPER_KEY in this code to find the correct place to provide that key..
# Please ensure that you have enabled the YouTube Data API for your project from your Google account.

    DEVELOPER_KEY = 'AIzaSyBN0zRiSDC_IdQrYWQaTcbCheyKLRopqOA'
    YOUTUBE_API_SERVICE_NAME = 'youtube'
    YOUTUBE_API_VERSION = 'v3'
    
    # This code collects only ids and snippets of videos 
    # (as chennels and playlists didn't include any suitable information for us)
    
    def youtube_search(searchItem, maxResults):
        #print("searchItem: \n" + searchItem[0] + "\n" + searchItem[1])
        
        #youtubeSearchlist = []
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    
        # Call the search.list method to retrieve results matching the specified query term (at the end of the code).
        search_response = youtube.search().list(q=searchItem, part='id,snippet', maxResults=maxResults).execute()
    
        videos = []     #  channels = []  #  playlists = []
    
        # Add each result to the appropriate list, and then display the lists of matching videos. (channels, and playlists)
        
    #print(str(search_response.get('pageInfo')), '\n\n')
    #print(search_response)
        
        info = search_response.get('pageInfo', [])
        total = info.get('totalResults')
        #print(total)
        for search_result in search_response.get("items", []):
            if search_result['id']['kind'] == 'youtube#video':
                  videos.append('%s' % (search_result['snippet']['title']))
                  
                  
            # Code below saved for later use
            #                              search_result['id']['videoId']))
            #      elif search_result['id']['kind'] == 'youtube#channel':
            #            channels.append('%s (%s)' % (search_result['snippet']['title'],
            #                                 search_result['id']['channelId']))
            #      elif search_result['id']['kind'] == 'youtube#playlist':
            #            playlists.append('%s (%s)' % (search_result['snippet']['title'],
            #                                  search_result['id']['playlistId']))
    
        # print ('Videos:\n')
        #print(videos)
             #   print ('Channels:\n', '\n'.join(channels), '\n')
             #   print ('Playlists:\n', '\n'.join(playlists), '\n')
        return videos    
    
    #sentence = "AI and humans have always been friendly."
    #PP.displayInsertSentencesLayout()
    #sentence = PP.getSentences()[0]
    videos1 = youtube_search(sentence1, 10)
    #sentence = PP.getSentences()[1]
    videos2 = youtube_search(sentence2, 10)
    


    videos=[videos1,videos2]
    #print ('Videos:\n')
    
    #print(videos)
    #for youtube_doc in sentence:
    #    try:
    #                        # youtube_search(args)
    #                        # searchItem = "How to build up super snippets for new web content?"
    #        maxResults = 10
    #        videos = youtube_search(youtube_doc, maxResults)
    #    except HttpError as e:
    #        print ('An HTTP error %d occurred:\n%s' % (e.resp.status, e.content))
    
    # sentence = ["AI and humans have always been friendly.","AI is our friend and it has been friendly."
    
    
    
    
    #?+++## youtbeSearchSnippetlist form should be same with googlesearch: [10 snippets for sentence1,10 snippetsfor sentence2]
    
    # sentence = [S1,S2] ################## this
    
    #------------------- Text processing-------------------------#
    
    # Text processing
    import nltk
    import string
    import math 
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem.porter import PorterStemmer
    import re
    import sys
    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)
    
    pd.set_option('display.max_columns', None)      # or 1000
    pd.set_option('display.max_rows', None)         # or 1000
    pd.set_option('display.max_colwidth', -1)       # or 199
    
    
    low_doc = []
    for item in videos:
        low_documents = []
        for document in item:
            low_documents.append(document.lower())
        data_low = pd.DataFrame(low_documents)
        data_low.columns = ['Lower case sentence']
        #print(data_low) 
        low_doc.append(low_documents)
    
    #print("low_doc:")
    #print(low_doc)
    #print(len(low_doc)) 
        
    # tokenization by split # Sentences Tokenized into Words - split by whitespace
    sentence_doc = []
    for item in low_doc:
        sentences_documents = []
    #document_counter = 0
        for document in item:
            sentences_documents.append(document.split())
        printableList1 = []
        for sentence1 in sentences_documents:
            sentence1AsString = ''
            for idx1, aWord1 in enumerate(sentence1):        
                if idx1 == len(sentence1) - 1:
                    sentence1AsString = sentence1AsString + aWord1
                else:
                    str1 = aWord1 + ', '
                    sentence1AsString = sentence1AsString + str1
            printableList1.append(sentence1AsString)
        data_sentences1 = pd.DataFrame(printableList1)
        data_sentences1.columns = ['Sentence tokenized into words   - string form and comma separated for display']
        #print(data_sentences1)
        sentence_doc.append(printableList1)
    #print(len(sentence_doc))    
        
    # change compound words to separate words ie. 'conditional-statements' -> 'conditional', 'statements' 
    #print("\n" 'Single words' "\n")
    separate_doc = []
    for itemlist in sentence_doc:
        single_word_documents=[]
        for sentence_words in itemlist:
            single_word_list = []
            #for word in sentence_words:
            regex = re.compile("[-_]")
            trimmed = regex.sub('', sentence_words)
            #print(trimmed)
            separate = trimmed.split( )
            for item in separate:
                single_word_list.append(item)        
            single_word_documents.append(single_word_list)
            #print(single_word_documents)
            printableList2 = []
            for sentence2 in single_word_documents:
                sentence2AsString = ''
                for idx2, aWord2 in enumerate(sentence2):        
                    if idx2 == len(sentence2) - 1:
                        sentence2AsString = sentence2AsString + aWord2
                    else:
                        str2 = aWord2 + ', '
                        sentence2AsString = sentence2AsString + str2
                printableList2.append(sentence2AsString)
        data_sentences2 = pd.DataFrame(printableList2)
        data_sentences2.columns = ['Single words   - string form and comma separated for display']
        #print(data_sentences2)     
        separate_doc.append(printableList2)
    #print(len(separate_doc)) 
    
    # remove all tokens that are not alphabetic #############
    #print("\n" 'Tokenized with alphabetic chars only' "\n")
    alpha_doc = []
    for itemlist in separate_doc:
        alpha_documents = []
        for single_word_sentence in itemlist:
            cleaned_list = []
            #for single_word in single_word_sentence:
            regex = re.compile('[^a-zA-Z]')
            #First parameter is the replacement, second parameter is your input string
            nonAlphaRemoved = regex.sub(' ', single_word_sentence)
            # add string to list only if it has content
            if nonAlphaRemoved:
                cleaned_list.append(nonAlphaRemoved)
            alpha_documents.append(cleaned_list)
    #print(alpha_documents)
    
            printableList3 = []
            for sentence3 in alpha_documents:
                sentence3AsString = ''
                for idx3, aWord3 in enumerate(sentence3):        
                    if idx3 == len(sentence3) - 1:
                        sentence3AsString = sentence3AsString + aWord3
                    else:
                        str3 = aWord3 + ', '
                        sentence3AsString = sentence3AsString + str3
                printableList3.append(sentence3AsString)
        data_sentences3 = pd.DataFrame(printableList3)
        data_sentences3.columns = ['Tokenized with alphabetic chars only   - string form and comma separated for display']
        #print(data_sentences3) 
        alpha_doc.append(printableList3)
    #print(len(alpha_doc))
    
    # filter out stopwords ###########
    #print("\n" 'English stopwords filtered tokens' "\n")
    stopwds_filtered = []
    english_stop_words = set(stopwords.words('english'))
    #print(english_stop_words)
    
    def remove_stopwords(word_list):
        processed_word_list = []
        for word in word_list:
            if word not in english_stop_words:
                processed_word_list.append(word)
        return processed_word_list
    #word_list=["I","dont","know","where","are","you"] 
    #print(remove_stopwords(word_list))
    
    for itemlist in alpha_doc:
        stop_filtered_tokens = []  
        for item in itemlist:
            stop_filtered_tokens.append(remove_stopwords(item.split()))
            #print(stop_filtered_tokens)  
            printableList4 = []
            for sentence4 in stop_filtered_tokens:
                sentence4AsString = ''
                for idx4, aWord4 in enumerate(sentence4):        
                    if idx4 == len(sentence4) - 1:
                        sentence4AsString = sentence4AsString + aWord4
                    else:
                        str4 = aWord4 + ', '
                        sentence4AsString = sentence4AsString + str4
                printableList4.append(sentence4AsString)
        data_sentences4 = pd.DataFrame(printableList4)
        # data_sentences4.columns = ['English stopwords filtered tokens   - comma separated for display']
        # print(data_sentences4)  
        stopwds_filtered.append(printableList4)
    
    # tokenization by PorterStemmer ############
    #print("\n" 'Word Stemming by PorterStemmer' "\n")
    PS = PorterStemmer()
    stem_doc = []
    for itemlist in stopwds_filtered:
        porter_doc = []
        for item in itemlist:
            #print(item.split())
            item=item.replace(',','')
            newitem=item.split()
            # print(newitem)
            item_doc=[]
            for word in newitem:
                porter_doc.append(PS.stem(word))
            #print(porter_doc)
        item_doc.append(porter_doc)
        stem_doc.append(item_doc) 
    #print(stem_doc)
    #----------- define jaccard similarity for python-------------#
    
    # define jaccard similarity for python #
    def jaccard_similarity(query, jdoc):
        intersection = set(query).intersection(set(jdoc))
        union = set(query).union(set(jdoc))
        return len(intersection)/len(union)
    
    def listToString (sourceList):
        subListAsString = ''    
        for listIndex, listWord in enumerate(sourceList):        
            if listIndex == len(sourceList) - 1:
                subListAsString = subListAsString + listWord
            else:
                strWithComma = listWord + ','
                subListAsString = subListAsString + strWithComma        
        return subListAsString
    
    # convert list string to one string list
    wdStr=[]
    #print("stopwds_filtered")
    #print(stopwds_filtered)
    for itemlist in stopwds_filtered:
        itemlistStr=listToString(itemlist)
        wdStr.append(itemlistStr.split(','))
    
    j_result1 = jaccard_similarity(wdStr[0], wdStr[1]) ################# this
    print('j_result1: wdstr')
    print(j_result1)
    #print('JSim_stopwds_filtered=',j_result1) # 0.11877394636015326
    #j_result2 = jaccard_similarity(stem_doc[0][0], stem_doc[1][0]) ############# this.....
    #print('j_result2:stem_doc')
    #print(j_result2)
    #print('JSim_stem=',j_result2)             # 0.11904761904761904
    
    
    
    #-----------------Wordnet Similarity for Snippetlist---------------------------------#
    
    from nltk import word_tokenize, pos_tag
    from nltk.corpus import wordnet as wn
    import numpy as np
    
    def penn_to_wn(tag):
        """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
        if tag.startswith('N'):
            return 'n'
     
        if tag.startswith('V'):
            return 'v'
     
        if tag.startswith('J'):
            return 'a'
     
        if tag.startswith('R'):
            return 'r'
     
        return None
     
    def tagged_to_synset(word, tag):
        wn_tag = penn_to_wn(tag)
        if wn_tag is None:
            return None
     
        try:
            return wn.synsets(word, wn_tag)[0]
        except:
            return None
     
    def Wn_similarity(sentence1, sentence2):
        """ compute the sentence similarity using Wordnet """
        # Tokenize and tag
        #print("sentence1:")
        #print(sentence1)
        #print("sentence2:")
        #print(sentence2)
    
    
        sentence1 = pos_tag(sentence1)
        sentence2 = pos_tag(sentence2)
     
        # Get the synsets for the tagged words
        synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
        synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
     
        # Filter out the Nones
        synsets1 = [ss for ss in synsets1 if ss]
        synsets2 = [ss for ss in synsets2 if ss]
     
        #print("synsets1:")
        #print(synsets1)
        #print("synsets2:")
        #print(synsets2)
    
        score, count = 0.0, 0
 
        # For each word in the first sentence
        best_scores = [0.0]
        for ss1 in synsets1:
            for ss2 in synsets2:
                eachscore=ss1.path_similarity(ss2)
                if eachscore is not None:
                    best_scores.append(eachscore)
            max1=max(best_scores)
            if best_scores is not None:
                score = score + max1
            if max1 is not 0.0:
                count = count + 1  
        #print(score/count)      
        # Average the values
        return score / count     
    
    #print(wdStr)
    # Snippetlist Wn_similarity
    wn_result1 = Wn_similarity(wdStr[0], wdStr[1])
    print('wn_result1:')
    print(wn_result1)
    #print('wnSim_stopwdStr=',wn_result1) # 0.3333333333333333
    #wn_result2 = Wn_similarity(stem_doc[0][0], stem_doc[1][0])
    #print('wn_result2:')
    #print(wn_result2)
    #print('wnSim_stemDoc=',wn_result2)   # 0.10194849748482962
    
  
    
    #------------------ExpandJaccardSimilarity--------------------------#
    
    #Snippetlist_ExpandJaccardSimilarity_Sicily
    from nltk.corpus import wordnet as wn
    import itertools
    from itertools import chain
    # we use stopwds_filtered -list of word lists that is defined above
    # as seed for wordnet
    
    def get_expandwds(word_list):
        expandedList=[]
        for ss in word_list: 
            hyper_list = []
            hypo_list = []
            new_list = []      
            mySynSets = wn.synsets(ss.strip()) #'ss.strip()'--->remove whitespace in the string
            for i,j in enumerate(mySynSets):    
            #print(i,j.name())
                hyper_list.append(list(chain(*[i.lemma_names() for i in j.hypernyms()])))
            #print(hyper_list)
                hypo_list.append(list(chain(*[i.lemma_names() for i in j.hyponyms()])))
            #print(hypo_list)
                new_list = hypo_list+hyper_list
                new_list=[item for item in new_list if item]
            #print(new_list)
            expandedList.append(new_list)
        # convert '[]' to '()'
        ExpandSentence=[]
        for sentence in expandedList:
            newitem=[]
            for eachitem in sentence:
                newitem.append(tuple(eachitem))
            ExpandSentence.append(tuple(newitem))
        #print(ExpandSentence_2)
        return ExpandSentence
    
    ej_result1 = jaccard_similarity(get_expandwds(wdStr[0]), get_expandwds(wdStr[1])) ########### this  hponyms
    print('ExpandJaccardSim_snippetlist=',ej_result1) # 0.15135135135135136
    
    #ej_result2 = jaccard_similarity(get_expandwds(word_tokenize(sentence[0])),get_expandwds(word_tokenize(sentence[1]))) ########### this... no need
    #print('ExpandJaccardSim_sentence=',ej_result2)    # 0.4444444444444444

    import gensim
    from gensim.corpora import MmCorpus
    from gensim import corpora, models, similarities
    from gensim.utils import simple_preprocess
    from gensim.models import LsiModel
    from math import *
    import numpy as np
    from nltk.corpus import stopwords

    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    STOPWORDS=set(stopwords.words('english'))

    def tokenize(text):
        return [token for token in gensim.utils.simple_preprocess(text) if token not in STOPWORDS]

    id2word = gensim.corpora.Dictionary.load_from_text ('./wikiresults/results_wordids.txt.bz2')
    #mm = gensim.corpora.MmCorpus('./results/results_tfidf.mm')-22.4GB
    #print(mm)

    # train LSI model
    # model_lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=400)
    # model_lsi.save('./wikiresult/model_lsi.model')

    # load LSI model

    model_lsi = LsiModel.load('./wikiresults/lsi.lsi_model')

    # doc to bag of words vector
    def get_vector(sentence):
        """ compute lsivectors using LSI model """
        vec_bow = id2word.doc2bow(tokenize(sentence)) #or item.lower().split()
        return vec_bow

    # cosine similarity
    def cosine(v1,v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)

    # get lsi vectors to compute sentence similarity
    def get_sentence_sim(S1,S2):
        vec_bow1=get_vector(S1)
        vec_bow2=get_vector(S2)
        vec_lsi1 = [val for idx,val in model_lsi[vec_bow1]]
        vec_lsi2 = [val for idx,val in model_lsi[vec_bow2]]
        return cosine(vec_lsi1,vec_lsi2)
        
    wiki_sim=get_sentence_sim(sentence1,sentence2)
    print('wikipedia similarity:',wiki_sim)
