import streamlit as st
import nltk
import numpy as np
import string
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
# Sample knowledge base with website references
faq_data = {
    "hello": "Hello! How can I assist you today?",
    "course details": "The offered courses are MCA, MSC, BTech, and more. Check the website for details! <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "admission process": "Admissions open in June and December. Visit the official site to apply. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "fees": "Course fees vary. Please check our official website for the latest fee structure. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "timetable": "The timetable is available on the student portal. Log in to check your schedule. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "class timings": "Classes generally run from 9 AM to 5 PM. Check the timetable for details. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "thank you": "You're welcome! Happy to help.",
    "exam schedule": "The exam schedule is available on the student portal. Log in to check the dates. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "hostel details": "Hostel facilities are available on campus. Visit the student portal for more information on availability and accommodation options. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "library hours": "The library is open from 8 AM to 10 PM on weekdays and 9 AM to 5 PM on weekends. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",  # Added specific response for library hours
    "scholarship": "Scholarships are available for eligible students. Please check the official website for details on eligibility and how to apply. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "internships": "We offer various internship programs in collaboration with top companies. Check the student portal for the latest opportunities. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "placement details": "Our placement cell works to place students in leading companies. Visit the career services section of the student portal for more details. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "academic calendar": "The academic calendar is available on the student portal. Please log in to check important dates and holidays. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "contact faculty": "You can contact faculty through the student portal or via email. The contact details of all faculty members are available there. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "degree certification": "Once you complete your course, you can apply for your degree certification via the student portal. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "counseling services": "Counseling services are available for students. Please check the student portal for appointment details. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "events": "Various events, workshops, and seminars are organized throughout the year. Check the events section of the student portal for upcoming activities. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "sports facilities": "Our campus has a fully equipped sports complex with various indoor and outdoor facilities. Please visit the sports section of the student portal for more information. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "transportation": "We provide transportation services for students. Visit the transport section of the student portal for bus schedules and routes. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "attendance policy": "The attendance policy requires a minimum of 75% attendance to appear for exams. Check the academic regulations on the student portal for more details. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "course prerequisites": "Some courses have prerequisites. Check the course catalog on the student portal for detailed information. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "student clubs": "We have a variety of student clubs including cultural, technical, and sports clubs. You can join any club through the student portal. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>",
    "career counseling": "Career counseling services are available for students. Please book an appointment through the student portal. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>"
}

# Preprocessing function
def preprocess_text(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return " ".join(tokens)

# Preprocess the FAQ data
processed_questions = [preprocess_text(q) for q in faq_data.keys()]
answers = list(faq_data.values())

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)

def chatbot_response(user_input):
    user_input = preprocess_text(user_input)
    user_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    best_match_idx = np.argmax(similarities)
    
    if similarities[0][best_match_idx] > 0.2:
        return answers[best_match_idx]
    else:
        return "I'm sorry, I don't have information on that. Please check the student portal. <a href='https://www.bbau.ac.in/' target='_blank'>Visit Website</a>"

# Streamlit interface with modern UI, logo, and buttons
def main():
    st.title("Student Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Start the chat session
    if st.button("Start Chat", key="start"):
        st.session_state.chat_history.append("Chatbot: Hello! How can I assist you today?")
        st.write("Chat has started!")

    # Display chat history with clickable links
    chatbox = st.container()
    with chatbox:
        for message in st.session_state.chat_history:
            if message.startswith("You:"):
                st.markdown(f'**You**: {message[4:]}')
            else:
                st.markdown(f'**Chatbot**: {message[9:]}', unsafe_allow_html=True)

    # Input and buttons for Send and End below chat messages
    user_input = st.text_input("You:", "")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("Send", key="send"):
            if user_input:
                st.session_state.chat_history.append(f"You: {user_input}")
                response = chatbot_response(user_input)
                st.session_state.chat_history.append(f"Chatbot: {response}")
                
    with col2:
        if st.button("End Chat", key="end"):
            st.session_state.chat_history = []
            st.write("Chat ended. Have a nice day!")

if __name__ == "__main__":
    main()
