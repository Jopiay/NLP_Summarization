import string
import re
from collections import defaultdict
import math
import numpy
np = numpy
stop_words = [
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
                'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
                'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
                'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
                'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 
                'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 
                'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']




parag = "This is a sentence. Here's another one. And a third sentence."




def remove_word(sentence, word):#using regular expression to replace a word with empty string
    sentence = re.sub(r'\b' + re.escape(word) + r'\b', '', sentence)
    return sentence


##split and tokenize the document into sentences##
def Tokenization(paragr):
    paragr = paragr.lower()
    sentences = re.split(r'(?<=\.)\s+', paragr.strip())
    cleaned_sentence = []

    for sentence in sentences:#removing stop-words
        for word in stop_words:
                sentence = remove_word(sentence, word)

        cleaned_sentence.append(sentence.strip())
    
    return cleaned_sentence

##Vectorizing the sentences
##one of the many ways is to use TF-IDF
def TF(sentences = Tokenization(parag)):
    tf = []
    for sentence in sentences:
        wrd_count = defaultdict(int)
        words = sentence.split()
        for word in words:
            wrd_count[word] += 1
        tf.append(wrd_count)
    return tf

def IDF(sentences = Tokenization(parag)):
    idf = defaultdict(int)
    num_sentences = len(sentences)
    for sentence in sentences:
        words = set(sentence.split())
        for word in words:
            idf[word] += 1
    for word in idf:
        idf[word] = math.log(num_sentences/ (1 + idf[word]) + 1)
    return idf
              
def tf_idf(tf  , idf ):
    term_freq_inverse_term_freq = []
    
    for sentence_tf in tf:
        tfidf_sentence = {}
        for word, count in sentence_tf.items():
            tfidf_sentence[word] = count * idf[word]
        term_freq_inverse_term_freq.append(tfidf_sentence)

    return term_freq_inverse_term_freq


##creating consine similarity to see how similar 2 vectors are to each other
def cos_simi(vec1, vec2):
    dot_product = sum([vec1.get(word, 0) * vec2.get(word, 0) for word in set(vec1) | set(vec2)])
    norm_a = math.sqrt(sum([val**2 for val in vec1.values()]))
    norm_b = math.sqrt(sum([val**2 for val in vec2.values()]))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

    
def cos_simi_matrix(tfidf):
    n = len(tfidf)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = cos_simi(tfidf[i], tfidf[j])
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim
    return similarity_matrix






def run():
    
    sentences = Tokenization(parag)  # Tokenize the paragraph
    tf_result = TF(sentences)  # Calculate term frequencies
    idf_result = IDF(sentences)  # Calculate inverse document frequencies

    # Print the tokenized sentences
    print("Tokenized Sentences:", sentences)
    
    # Print the term frequency result
    print("TF Result:", tf_result)
    
    # Print the inverse document frequency result
    print("IDF Result:", idf_result)
    
    # Print the final TF-IDF result
    print("TF-IDF Result:", tf_idf(tf_result, idf_result))



if  __name__ == '__main__' :
    run()