import streamlit as st

# Model
import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
alay_dict = pd.read_csv('./input/new_kamusalay.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original', 1: 'replacement'})

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub('\n',' ',text) # Remove every '\n'
    text = re.sub('rt',' ',text) # Remove every retweet symbol
    text = re.sub('user',' ',text) # Remove every username
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL
    text = re.sub('  +', ' ', text) # Remove extra spaces
    return text
    
def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) 
    return text

alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

def remove_stopword(text):
    list_stopwords = stopwords.words('indonesian')
    text = ' '.join(['' if word in list_stopwords else word for word in text.split(' ')])

    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = text.strip()
    return text

def stemming(text):
    return stemmer.stem(text)

def preprocess(text):
    text = lowercase(text) # 1
    text = remove_nonaplhanumeric(text) # 2
    text = remove_unnecessary_char(text) # 2
    text = normalize_alay(text) # 3
    text = stemming(text) # 4
    text = remove_stopword(text) # 5
    return text

# Load model
import joblib

loaded_model = joblib.load("model_rf.pkl")

# Load TFIDF
# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

data_clean = pd.read_csv("./preprocessed_indonesian_toxic_tweet.csv", encoding='latin-1')

tf = TfidfVectorizer()
text_tf = tf.fit_transform(data_clean["Tweet"].astype('U'))

st.sidebar.subheader('About the App')
st.sidebar.write('Text Classification App with Streamlit using a trained Random Forest model')
st.sidebar.write("This is just a small text classification app. Don't fret if the prediction is not correct or if it is not what you expected, the model is not perfect.")
st.sidebar.write("There is no provision for neutral text, yet...")


#start the user interface
st.title("Text Classification App")
st.write("Type in your text below and don't forget to press the enter button before clicking/pressing the 'Classify' button")

my_text = st.text_input("Enter the text you want to classify", "Change this...", max_chars=100, key='to_classify')

if st.button('Classify', key='classify_button'):
    preprocess_text = preprocess(my_text)
    feature = tf.transform([preprocess_text])

    p = loaded_model.predict(feature)

    if p[0] > 0:
        # the predicted class is 1
        st.write(f"Your input text contains hs")
    else:
# otherwise the predicted class is 0
        st.write(f"Your input text doesn't contain hs")
