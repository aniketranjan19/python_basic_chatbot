# importing necessary modules
import nltk
import string
import random

# code for opening dataset
f = open('dataset.txt', 'r', errors='ignore')
file = f.read()

file = file.lower()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
sentence_tokens = nltk.sent_tokenize(file)
word_tokens = nltk.word_tokenize(file)

lemer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemer.lemmatize(token) for token in tokens]


remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))


greeting_inputs = ('hi', 'hello', 'hey', 'how are you?')
greeting_responses = ('Hii', 'Hey there!', 'hello', 'Nice to meet you')


def greet(sentence):
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def response(user_response):
    robot_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robot_response = robot_response + "I'm sorry, Unable to understand"
        return robot_response
    else:
        robot_response = robot_response + sentence_tokens[idx]
        return robot_response


flag = True
print('*****  For ending the conversation type --> bye  *****')
print()
print("Hello, I'm chatbot...")

while flag == True:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thankyou' or user_response == 'thanks':
            flag = False
            print('Bot: You are Welcome...')
        else:
            if greet(user_response) is not None:
                print('Bot: ' + greet(user_response))
            else:
                sentence_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print('Bot: ', end='')
                print(response(user_response))
                sentence_tokens.remove(user_response)
    else:
        flag = False
        print('Bot: Bye Bye!')