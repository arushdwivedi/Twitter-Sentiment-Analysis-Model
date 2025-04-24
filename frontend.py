
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
port_stem=PorterStemmer()
import nltk
nltk.download('stopwords')
import re
import streamlit as st
loadedmodel=pickle.load(open('trainedmodel.sav','rb'))


def stemming(content):
    stemmed_content= re.sub('[^a-zA-Z]', ' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)

    return stemmed_content

sampletweet = ["A person who loves you, will treat you well. You don't beg to be treated well."]
sampletweet = [stemming(tweet) for tweet in sampletweet] 

vectorizer2=pickle.load(open('vectorizer.sav','rb'))
sampletweet=vectorizer2.transform(sampletweet)

def predict(sampletweet):
    sampletweetpred= loadedmodel.predict(sampletweet)
    if sampletweetpred==1:
        st.text('positive tweet')
    else:
        st.text('negative tweet')

def main():
    st.title("Tweet Sentiment Analysis")
    st.markdown("""Tech Stack Overview: Tweet Sentiment Analysis

Frontend:
• Streamlit
  Used to build a simple and interactive web interface where users can input tweets and receive real-time sentiment predictions.

Backend & Logic:
• Python
  Core language used for integrating components, preprocessing text, and handling model inference.
• NLTK (Natural Language Toolkit)
  Utilized for natural language preprocessing:
  - Removal of stopwords
  - Word stemming using PorterStemmer
  - Cleaning text with regular expressions
• Scikit-learn
  - Vectorization: Converts text into numerical form using TF-IDF or CountVectorizer
  - Model: Trained using a machine learning algorithm (e.g., Logistic Regression, SVM) to classify tweet sentiment

Model & Data:
• Pickle (.sav files)
  Used to load:
  - The trained machine learning model
  - The vectorizer used during training

Deployment:
• Local Deployment with Streamlit
  Enables fast prototyping and testing of the app in a browser-based interface.
""")
    st.text("Enter the tweet to be analysed:")
    
    input_tweet = st.text_input("Tweet", "Type Here")

    if input_tweet and input_tweet != "Type Here":
        # Preprocess
        cleaned_tweet = stemming(input_tweet)
        
        # Vectorize
        vectorized_tweet = vectorizer2.transform([cleaned_tweet])  # Must be list format
        
        # Predict
        predict(vectorized_tweet)


if __name__ == '__main__':
    main()



