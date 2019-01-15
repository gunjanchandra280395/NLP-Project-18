This document gives instructions how to use Wikipedia dump files to conduct sentence similarity and search similar documents from Wikipedia.

1. Data preperation

[1] Download Wikipedia dump files (enwiki-latest-pages-articles.xml.bz2) from:
  	http://download.wikimedia.org/enwiki/ 
    The size of XML files is about 14GB.

[2] Parse wikipedia dump files
    'wikiparse.bat' can be used to parse the dump files.
    The original source code is from TOPIC MODEL gensim:https://radimrehurek.com/gensim/wiki.html#latent-semantic-analysis

[3] The parse process may need around 14hrs, which depends on your computer. Need 35GB to store the parsed documents. The parsed documents includde:
    * 'OUTPUT_PREFIX_wordids.txt': mapping between words and their integer ids
    * 'OUTPUT_PREFIX_bow.mm': bag-of-words (word counts) representation, in Matrix format
    * 'OUTPUT_PREFIX_tfidf.mm': TF-IDF representation
    * 'OUTPUT_PREFIX.tfidf_model': TF-IDF model dump

2. Train LSI model

[1] tool: gensim library, python.

[2] The model training refers to:
    https://radimrehurek.com/gensim/wiki.html#latent-semantic-analysis

[3] The code can be found in 'wiki_sentencesimilarity.py' file.

3. Compute sentence similarity

[1] Run 'wiki_sentencesimilarity.py' to get any input two sentences similarity.

[2] Sentence similarity is computed by comparing the LSI vectors after converting to LSI space (cosine similarity).

4. Search similar documents associated to the input sentence from Wikipedia dump files

[1] Run 'wiki_simdoc.py' to search similar documents from Wikipedia dump files

[2] The method is to use "Index" from gensim--->return (doc no.,cosine similarity)


Note: Some files are too big to upload to github. You need to do the above process to get the results.



