# Chatbot Using Logistic Regression  

![Chatbot](https://img.shields.io/badge/Chatbot-Streamlit-blue.svg)  
A simple and interactive chatbot built using **Logistic Regression**, **TF-IDF Vectorization**, and **Streamlit** for the interface. This chatbot uses predefined intents and patterns to provide relevant responses.  

## Features  
- **Interactive UI**: A clean and simple chatbot interface using Streamlit.  
- **Intent Classification**: Classifies user input into predefined intents using Logistic Regression.  
- **Customizable Responses**: Easily extend the chatbot with new intents and patterns.  
- **Educational Purpose**: Great for understanding the basics of NLP, intent classification, and deploying with Streamlit.  

---

## Installation  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/chatbot-logistic-regression.git
   cd chatbot-logistic-regression
   ```  

2. **Install Required Libraries**  
   Make sure you have Python installed, then run:  
   ```bash
   pip install -r requirements.txt
   ```  

3. **Download NLTK Data**  
   The chatbot requires the `punkt` tokenizer from NLTK:  
   ```python
   import nltk
   nltk.download('punkt')
   ```  

---

## Usage  

1. **Run the Chatbot**  
   Execute the following command in your terminal to launch the Streamlit app:  
   ```bash
   streamlit run chatbot_streamlit.py
   ```  

2. **Chat with the Bot**  
   - Type your message in the input box.  
   - Press the **Send** button to get a response.  
   - Type "quit" to end the conversation.  

---

## Example Intents  
Here are some example intents supported by the chatbot:  

| Intent   | Example Inputs                          | Example Responses                       |  
|----------|-----------------------------------------|-----------------------------------------|  
| Greeting | "Hi", "Hello", "What's up"             | "Hello!", "Hi there!"                  |  
| Goodbye  | "Bye", "See you later", "Take care"    | "Goodbye!", "Take care!"               |  
| Help     | "Can you help me?", "I need help"      | "Sure, what do you need help with?"    |  
| About    | "Who are you?", "What is your purpose" | "I am a chatbot. I can assist you."    |  

---

## Code Highlights  

1. **Intent Classification**  
   Uses `TF-IDF Vectorization` and `Logistic Regression` for text classification.  

2. **Dynamic Responses**  
   Selects a random response for each intent to make the chatbot feel more interactive.  

3. **Streamlit Interface**  
   Provides a lightweight, easy-to-use web interface for chatting with the bot.  

---

## File Structure  

```plaintext
.
├── chatbot_streamlit.py      # Main Python file for the chatbot
├── chatbot.ipynb             # Original Jupyter Notebook (optional)
├── requirements.txt          # Required Python libraries
├── nltk_data/                # Directory for NLTK data
└── README.md                 # Project documentation
```  

---

## Future Enhancements  

- Add support for **real-time APIs** (e.g., weather or news).  
- Enable a **persistent conversation history**.  
- Expand the **intent library** for broader functionality.  

---

## License  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  
