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




parag = """ 
            Technology has become an integral part of modern education, revolutionizing how students learn and teachers instruct. From interactive whiteboards to online learning platforms, the tools available today have transformed traditional classrooms into dynamic environments that foster collaboration and innovation.

One of the most significant benefits of technology in education is accessibility. Online courses and digital resources allow students from remote or underserved areas to access quality education. Moreover, students with disabilities can use assistive technologies to overcome learning barriers, ensuring inclusivity in the learning process.

Another crucial advantage is personalized learning. Adaptive learning software tailors educational content to the needs of individual students, helping them learn at their own pace. This approach not only improves understanding but also boosts confidence and motivation.

Collaboration has also been enhanced by technology. Tools like Google Workspace, Microsoft Teams, and educational apps enable students to work together on projects, regardless of their physical location. Virtual classrooms and video conferencing ensure that learning continues uninterrupted, even in challenging times like the COVID-19 pandemic.

However, the integration of technology in education is not without challenges. One major issue is the digital divide—many students still lack access to devices or reliable internet connections, widening the gap between privileged and underserved communities. Cybersecurity and data privacy are additional concerns, as increased reliance on technology exposes educational institutions to potential threats.

Despite these challenges, the future of education undoubtedly lies in leveraging technology. Governments, educators, and technology providers must work together to address existing barriers and ensure that the benefits of technology are accessible to all.

In conclusion, technology has the potential to transform education for the better, making it more inclusive, engaging, and efficient. As we continue to innovate, it is crucial to ensure that no student is left behind in the pursuit of knowledge.


        """




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
##one of the different ways is to use TF-IDF
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


def most_similar(tfidf, sentences, k):
    simi_matrix = cos_simi_matrix(tfidf)
    row_sum = np.sum(simi_matrix, axis=1)
    sorted_indices = np.argsort(row_sum)[::-1]  # Sort indices by descending scores
    top_k_indices = sorted_indices[:k]  # Select top-k indices
    summary = [sentences[i] for i in top_k_indices]  # Fetch sentences
    return summary



def choose_k():
    k = input("Choose the number of sentences the summary is of: ")
    return int(k)  # Convert to integer

def run():
    sentences = Tokenization(parag)  # Tokenize the paragraph
    tf_result = TF(sentences)  # Calculate term frequencies
    idf_result = IDF(sentences)  # Calculate inverse document frequencies
    tfidf_result = tf_idf(tf_result, idf_result)
    
    # Number of sentences for the summary
    k = choose_k()
    
    # Generate and print summary
    summary = most_similar(tfidf_result, sentences, k)
    print("\nSummary:")
    print("\n".join(summary))
    



if __name__ == '__main__':
    run()


