import pandas as pd
import re
from transformers import BertTokenizer, TFBertForSequenceClassification, pipeline
import streamlit as st

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
loaded_model= TFBertForSequenceClassification.from_pretrained("tweet_sentiment_classifier_bert")
clf=pipeline("text-classification", model=loaded_model, tokenizer=tokenizer)

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)

def get_prediction(text):
    pred=clf.predict(text)[0]['label']
    if pred=='LABEL_0':
        result='negative'
    elif pred=='LABEL_2':
        result='positive'
    else:
        result='neutral'
    return(result)

def predict_sentiment(text):
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction

def main():
    #st.title("Tweet Classifier")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Tweet Classifier App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    tweet = st.text_input("tweet","Type Here")

    result=""
    if st.button("Predict"):
        result=get_prediction(clean_text(tweet))
    st.success('The output is {}'.format(result))

if __name__=='__main__':
    main()