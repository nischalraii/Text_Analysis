import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from transformers import pipeline
import torch


# Initialize the question-answering pipeline once
@st.cache_resource
def initialize_pipeline():
    return pipeline("question-answering", model="Ferreus/QA_model")


# Sidebar contents
with st.sidebar:
    st.title('Question Answering from given Context')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Transformers](https://huggingface.co/transformers/) library
    ''')
    add_vertical_space(5)
    st.write('Made by Nischal Raii')


def main():
    st.header("Below are the options")

    st.subheader("1. QA")
    st.subheader("2. Summarize")

    selected = st.selectbox('What do you want to do?',('1','2'),index=None,
    placeholder="Select contact method...",)
    st.write(f"Selected Option: {selected}")

    if selected == '1':
        st.header("Question Answer")
        # Initialize the pipeline
        question_answerer = initialize_pipeline()

        context = st.text_area('Give the Context:')

        if context:
            question = st.text_input('Ask a question about the context:')

            if question:
                st.write(f"**Question:** {question}")

                # Get the answer
                result = question_answerer(question=question, context=context)

                # Display the result
                st.write(f"**Answer:** {result['answer']}")
            else:
                st.write("Please enter a question.")
    elif selected == '2':
        st.header("Summarize")
        st.write("Summarize not supported yet")

    else:
        st.write("Please enter valid input")

if __name__ == '__main__':
    main()
