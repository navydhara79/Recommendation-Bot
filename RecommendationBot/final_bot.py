
import random
import json
import os
import re

#%%script false
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from os.path import join
from tqdm import tqdm

tqdm.pandas()

#rb_context

class Context(object):

    def __init__(self,name):
        self.lifespan = 2
        self.name = name
        self.active = False

    def activate_context(self):
        self.active = True

    def deactivate_context(self):
        self.active = False

        def decrease_lifespan(self):
            self.lifespan -= 1
            if self.lifespan==0:
                self.deactivate_context()

class FirstGreeting(Context):

    def __init__(self):
        self.lifespan = 1
        self.name = 'FirstGreeting'
        self.active = True

#rb_actions

'''This function masks the entities in user input, and updates the attributes dictionary'''
def getattributes(uinput, context, attributes):

    # Can use context to context specific attribute fetching
    # print("getattributes context ", context)
    if context.name.startswith('IntentComplete'):
        return attributes, uinput
    else:
        # Code can be optimised here, loading the same files each time suboptimal
        files = os.listdir('./entities/')
        # Filtering dat files and extracting entity values inside the entities folder
        entities = {}
        for fil in files:
            if fil == ".ipynb_checkpoints":
                continue
            lines = open('./entities/'+fil).readlines()
            for i, line in enumerate(lines):
                lines[i] = line[:-1]
            entities[fil[:-4]] = '|'.join(lines)

        # Extract entity and update it in attributes dict
        for entity in entities:
            for i in entities[entity].split('|'):
                if i.lower() in uinput.lower():
                    attributes[entity] = i

        # Masking the entity values $ sign
        for entity in entities:
            uinput = re.sub(entities[entity], r'$'+entity, uinput, flags=re.IGNORECASE)


        return attributes, uinput

'''Spellcheck and entity extraction functions go here'''
def input_processor(user_input, context, attributes, intent):

    # uinput = TextBlob(user_input).correct().string

    # update the attributes, abstract over the entities in user input
    attributes, cleaned_input = getattributes(user_input, context, attributes)

    return attributes, cleaned_input

'''This function is used to classify the intent'''
def intentIdentifier(clean_input, context, current_intent):
    
    clean_input = clean_input.lower()
    # print("intentIdentifier - clean_input ", clean_input)
    '''Word Embedding using Bag of Words'''
    p = pd.read_csv('corpus.csv')
    df = pd.DataFrame(p)
    df['intents_cleaned'] = df['intents'].progress_apply(text_normalization)

    vectorizer = CountVectorizer()
    train_features = vectorizer.fit_transform(df['intents_cleaned'])
    train_features = train_features.toarray()

    vocab = vectorizer.get_feature_names()
    df_bow = pd.DataFrame(train_features, columns=vocab)  # list of words
    
    lemmatizer = WordNetLemmatizer()
    stop=stopwords.words('english')

    query = clean_input
    print(query)
    
    q = []  
    a = query.split()
    for i in a:
        if i in stop:
            continue
        else:
            q.append(i)
        b = " ".join(q)

    ql = text_normalization(b)
    q_bow = vectorizer.transform([ql]).toarray()

    '''TODO : YOUR CODE HERE TO CLASSIFY THE INTENT'''
    # Scoring Algorithm, can be changed.

    df['classes'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1]
    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(train_features, df['classes'])
    pred = svm.predict(q_bow)
    print(pred)
    dfp = pd.DataFrame(pred)
    print(dfp)
    #scores = ngrammatch(clean_input)
    # choosing here the intent with the highest score
    #scores = sorted_by_second = sorted(scores, key=lambda tup: tup[1])
    # print('intentIdentifier - scores ', scores)

    if current_intent is None:
        if dfp[0][0] == 0:
            current_intent = loadIntent('params/newparams.cfg', 'CabType')
        elif dfp[0][0]== 1:
            current_intent = loadIntent('params/newparams.cfg', 'MovieRecommend')
        print("intentIdentifier - current_intent ", current_intent.name)
        return current_intent
    else:
        # If current intent is not none, stick with the ongoing intent
        return current_intent

def text_normalization(sentence):
        lemmatizer = WordNetLemmatizer()
        stop=set(stopwords.words('english'))
        sentence = str(sentence).lower()
        removed_punctuation = re.sub(r'[^a-zA-Z]', ' ', sentence)
        tokens = nltk.word_tokenize(removed_punctuation)
        lemma = WordNetLemmatizer()
        tags_list = pos_tag(tokens, tagset=None)
        lemma_words = []
        for token, pos_token in tags_list:
            if pos_token.startswith('v'):
                pos_val = 'v'
            elif pos_token.startswith('J'):
                pos_val = 'a'
            elif pos_token.startswith('R'):
                pos_val = 'r'
            else:
                pos_val = 'n'
            lemma_token = lemma.lemmatize(token, pos_val)
            lemma_words.append(lemma_token)
        return " ".join(lemma_words)
        removed_markup = BeautifulSoup(sentence, 'html.parser').text

        tokens = removed_punctuation.lower().split()
        removed_stopwords = [w for w in tokens if w not in stop]
        lemmatized = [lemmatizer.lemmatize(w) for w in removed_stopwords]
        return ' '.join(lemmatized)

'''Collects attributes pertaining to the current intent'''
def check_required_params(current_intent, attributes, context):

    for para in current_intent.params:
        if para.required == 'True':
            if para.name not in attributes:
                print('checking')
                # Example of where the context is born
                # if para.name=='RegNo':
                    # context = GetRegNo()
                # returning a random prompt from available choices.
                return random.choice(para.prompts), context

    return None, context

def check_actions(current_intent, attributes, context):
    '''This function performs the action for the intent
    as mentioned in the intent config file'''
    '''Performs actions pertaining to current intent
    for action in current_intent.actions:
        if action.contexts_satisfied(active_contexts):
            return perform_action()
    '''

    context = IntentComplete()
    # return 'action: ' + current_intent.action, context
    return current_intent.action, context

def loadIntent(path, intent):
    with open(path) as file_intent:
        dat = json.load(file_intent)
        intent = dat[intent]
        return Intent(intent['intentname'], intent['parameters'], intent['actions'])

#rb_intents

class Intent(object):
    # intent name, parameters and actions
    def __init__(self, name, params, action):
        self.name = name
        self.action = action
        self.params = []
        for param in params:
            # print param['required']
            self.params += [Parameter(param)]

class IntentComplete(Context):
    def __init__(self):
        self.lifespan = 1
        self.name = 'IntentComplete'
        self.active = True

class Parameter():
    def __init__(self, info):
        self.name = info['name']
        self.placeholder = info['placeholder']
        self.prompts = info['prompts']
        self.required = info['required']
        self.context = info['context']

#rb_session
class Session:
    '''Initialise a default session'''
    def __init__(self, attributes=None, active_contexts=[FirstGreeting(), IntentComplete() ]):

        # Active contexts not used yet, can use it to have multiple contexts
        self.active_contexts = active_contexts

        # Contexts are flags which control dialogue flow
        self.context = FirstGreeting()

        # Intent tracks the current state of dialogue
        #self.current_intent = First_Greeting()
        self.current_intent = None

        # attributes hold the information collected over the conversation
        self.attributes = {}

    '''Not used yet, but is intended to maintain active contexts'''
    def update_contexts(self):

        for context in self.active_contexts:
            if context.active:
                context.decrease_lifespan()

    '''Generate response to user input'''
    def reply(self, user_input):

        self.attributes, clean_input = input_processor(user_input, self.context, self.attributes, self.current_intent)

        self.current_intent = intentIdentifier(clean_input, self.context, self.current_intent)

        prompt, self.context = check_required_params(self.current_intent, self.attributes, self.context)

        # prompt is None means all parameters satisfied, perform the intent action
        if prompt is None:
            if self.context.name != 'IntentComplete':
                prompt, self.context = check_actions(self.current_intent, self.attributes, self.context)
                print("reply - prompt ", prompt, " context ", self.context)
                
                '''TODO : YOUR CODE HERE TO GET RECOMMENDATION BASED ON THE ENTFITY VALUES'''
                if(prompt == "CabType"):
                    print("You booked Ola Mini")
                elif(prompt == 'MovieRecommend'):
                    print("Watch Krishna and His Leela. Enjoy!")
                prompt = "Hi! How may I assist you?"

        # Resets the state after the Intent is complete
        if self.context.name == 'IntentComplete':
            self.attributes = {}
            self.context = FirstGreeting()
            self.current_intent = None

        return prompt

class GetRegNo(Context):

    def __init__(self):
        print('Hi')
        self.lifespan = 1
        self.name = 'GetRegNo'
        self.active = True

class SpellConformation(Context):

    def __init__(self,index,CorrectWord,user_input,context):
        self.lifespan = 1
        self.name = 'SpellConformation'
        self.active = True
        self.index = index
        self.correct = CorrectWord
        self.tobecorrected = user_input
        self.contexttobestored = context



#rb_main
session = Session()

print('BOT: Hi! How may I assist you?')

while True:
    
    inp = input('User: ')
    if inp == "exit" :
        break
    print('BOT:', session.reply(inp))
    
    