#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import nltk
import ssl
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL context issue for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')


# In[2]:


# Intents Data
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    }
]


# In[3]:


# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)


# In[4]:


# Define the chatbot function
def chatbot(input_text):
    input_text_vector = vectorizer.transform([input_text])
    tag = clf.predict(input_text_vector)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])


# In[5]:


# Streamlit App
def main():
    st.title("Chatbot App")
    st.write("### Hello! I'm a simple chatbot. Ask me anything! Type 'quit' to exit.")

    # User Input
    user_input = st.text_input("You:", key="user_input")
    if st.button("Send"):
        if user_input.strip() == "":
            st.write("**Chatbot**: Please type something to start chatting.")
        elif user_input.lower() == 'quit':
            st.write("**Chatbot**: Goodbye!")
        else:
            response = chatbot(user_input)
            st.write(f"**Chatbot**: {response}")


if __name__ == '__main__':
    main()

"""# First we have convert this Juypter Notebook into python file.
# get_ipython().run_line_magic('pip', 'install nbformat')
import nbformat
from nbconvert import PythonExporter

# Load the uploaded Jupyter Notebook file
notebook_path = "C:/Users/sachi/OneDrive/Desktop/Data Science/Chatbot-using-Logistic-Regression/chatbot.ipynb"

# Convert the notebook to a Python script
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook_content = f.read()

exporter = PythonExporter()
python_script, _ = exporter.from_notebook_node(nbformat.reads(notebook_content, as_version=4))

# Save the Python script
script_path = "C:?Users/sachi/OneDrive/Desktop/Data Science/Chatbot-using-Logistic-Regression/chatbot_streamlit.py"
with open(script_path, "w", encoding="utf-8") as f:
    f.write(python_script)

script_path
"""
