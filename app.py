import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

try:
    tk = pickle.load(open("vectorizer.pkl", 'rb'))
    model = pickle.load(open("model.pkl", 'rb'))
except FileNotFoundError:
    st.error("Model or Vectorizer files are missing. Please make sure 'vectorizer.pkl' and 'model.pkl' are available.")
    st.stop()

def transform_text(text):
    text = text.lower()  
    text = nltk.word_tokenize(text)  
    text = [i for i in text if i.isalnum()] 

    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    
    text = [ps.stem(i) for i in text]
    
    return " ".join(text)

st.title("SMS Spam Detection Model", anchor="title")


st.sidebar.markdown("""
    <div style="background-color:#f7f7f7; padding: 10px;">
        <h3 style="color:#0073e6;">How it works:</h3>
        <p style="color:#333;">This model predicts whether an SMS is <strong>Spam</strong> or <strong>Not Spam</strong>.</p>
        <ol style="color:#333;">
            <li>Enter the SMS text in the input field below.</li>
            <li>Click <strong>'Predict'</strong> to see the result.</li>
        </ol>
        <p style="color:#333;">The model classifies the message based on the content and known patterns of spam messages.</p>
    </div>
""", unsafe_allow_html=True)

input_sms = st.text_area("Enter the SMS", height=150, max_chars=300, placeholder="Type your SMS here...", key="sms_input")

if st.button('Clear Input', key='clear_input', help="Clear the input text"):
    st.experimental_rerun()

if st.button('Predict', key='predict', help="Click to classify the SMS"):

    if not input_sms.strip():
        st.warning("Please enter a message to classify.", icon="⚠️")
    else:
        with st.spinner('Classifying your message...'):
            transformed_sms = transform_text(input_sms)
            vector_input = tk.transform([transformed_sms])
            result = model.predict(vector_input)[0]

        
            if result == 1:
               
                st.markdown(
                    """
                    <style>
                        .main {
                            background-color: #ffcccc;
                            transition: background-color 1s ease;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                st.markdown("<h3 style='color: red;'>Prediction: <strong>Spam</strong></h3>", unsafe_allow_html=True)
                st.markdown("<p style='color: red;'>This message is classified as Spam based on its content. Spam messages often contain promotions or unwanted information.</p>", unsafe_allow_html=True)
            else:
                st.markdown(
                    """
                    <style>
                        .main {
                            background-color: #ccffcc;
                            transition: background-color 1s ease;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                st.markdown("<h3 style='color: green;'>Prediction: <strong>Not Spam</strong></h3>", unsafe_allow_html=True)
                st.markdown("<p style='color: green;'>This message is classified as Not Spam. It seems to be a legitimate message.</p>", unsafe_allow_html=True)
                
            st.subheader("Processed Text:")
            st.markdown(f"<p style='color:#666;'>{transformed_sms}</p>", unsafe_allow_html=True) 

st.sidebar.markdown("""
    <div style="background-color:#f7f7f7; padding: 10px;">
        <h4 style="color:#ff8c00;">Tips for Better Classification:</h4>
        <ul style="color:#333;">
            <li>Ensure the message is clear and in English.</li>
            <li>Avoid including too many abbreviations or informal language that may confuse the model.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #f1f1f1;
        }
        .css-1ojb9wa {
            background-color: #e0e0e0;
        }
        .css-1d91g2m {
            background-color: #f7f7f7;
        }
        h2 {
            color: #ff5733;
        }
        .stTextArea textarea {
            background-color: #fafafa;
        }
    </style>
    <div class="footer">
    <h2>Made with 🖤 by <b><I>SRIVARDHAN</I></b></h2>
""", unsafe_allow_html=True)
