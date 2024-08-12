import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import spacy
from heapq import nlargest
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud

# Initialize the question-answering pipeline once
@st.cache_resource
def initialize_pipeline():
    return pipeline("question-answering", model="Ferreus/QA_model")

def plot_similarity_matrix(similarity_matrix, num_sentences):
    """Calculate and visualize a similarity matrix for sentences."""
    labels = [f'Sentence {i + 1}' for i in range(num_sentences)]

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels, cmap='viridis', annot=True, ax=ax)
    ax.set_title('Sentence Similarity Matrix')
    ax.set_xlabel('Sentences')
    ax.set_ylabel('Sentences')

    st.pyplot(fig)


def plot_text_rank_scores(scores):
    """Visualize TextRank scores with generic sentence labels."""
    num_sentences = len(scores)
    labels = [f'Sentence {i + 1}' for i in range(num_sentences)]

    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]

    sns.barplot(x=sorted_scores, y=sorted_labels, palette='viridis', ax=ax)
    ax.set_title('TextRank Scores for Sentences')
    ax.set_xlabel('Score')
    ax.set_ylabel('Sentence')

    st.pyplot(fig)


def calculate_similarity_matrix(sentences):
    """Calculate a similarity matrix for sentences."""
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    similarity_matrix = np.dot(vectorizer, vectorizer.T).toarray()
    return similarity_matrix


def text_rank(sentences, similarity_matrix, damping=0.85, iterations=100):
    """Apply TextRank algorithm to the sentences."""
    n = len(sentences)
    scores = np.ones(n)
    for _ in range(iterations):
        new_scores = np.ones(n) * (1 - damping)
        for i in range(n):
            new_scores[i] += damping * np.sum(similarity_matrix[i] * scores)
        scores = new_scores
    return scores


def textrank_summarize(text, sentence_number=5):
    # Validate sentence_number
    if not isinstance(sentence_number, int) or sentence_number <= 0:
        raise ValueError("sentence_number must be a positive integer.")

    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    # Extract sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    num_sentences = len(sentences)

    if num_sentences == 0:
        return "No sentences to summarize."

    # Compute similarity matrix
    similarity_matrix = calculate_similarity_matrix(sentences)

    # Apply TextRank
    scores = text_rank(sentences, similarity_matrix)

    # Visualize Sentence Similarity Matrix
    st.write("Sentence Similarity Matrix:")
    plot_similarity_matrix(similarity_matrix, num_sentences)

    # Visualize TextRank Scores
    st.write("Text Rank Scores for Sentences:")
    plot_text_rank_scores(scores)

    # Rank sentences by score
    ranked_sentences = [sentences[i] for i in np.argsort(scores)[::-1]]

    # Extract the top sentences
    top_sentences = ranked_sentences[:sentence_number]

    # Join the top sentences and clean the result
    result = " ".join(top_sentences).replace('\n', ' ')

    return result

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


def plot_wordcloud(word_freq):
    # Preprocess word_freq to ensure no unexpected characters
    cleaned_word_freq = clean_word_freq(word_freq)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(cleaned_word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def clean_word_freq(word_freq):
    # Remove tokens that are not valid words
    return {word: freq for word, freq in word_freq.items() if word.isalpha()}

def word_freq_summarize(text, sentence_number=5):
    # Validate sentence_number
    if not isinstance(sentence_number, int) or sentence_number <= 0:
        raise ValueError("sentence_number must be a positive integer.")

    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    # Tokenize and compute word frequencies
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    word_freq = Counter(tokens)

    # Normalize word frequencies
    max_freq = max(word_freq.values(), default=1)  # Avoid division by zero
    word_freq = {word: freq / max_freq for word, freq in word_freq.items()}

    # Score sentences
    sent_score = {}
    for sent in doc.sents:
        sent_text = sent.text
        score = sum(word_freq.get(word.lower(), 0) for word in sent_text.split())
        if score > 0:
            sent_score[sent_text] = score

    # Extract the top sentences
    top_sentences = nlargest(sentence_number, sent_score, key=sent_score.get)

    # Join the top sentences and clean the result
    result = " ".join(top_sentences).replace('\n', ' ')

    return result , word_freq


def main():
    st.header("Below are the options")

    st.subheader("1. QA")
    st.subheader("2. Summarize")

    selected = st.selectbox('What do you want to do?',('1','2'),index=None,
    placeholder="Select an option",)
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
        text = st.text_area('Give the context for summarization:')
        sentence_number = st.number_input('How many sentences do you want to summarize to?', min_value=1,
                                          step=1)

        if text:
            selected_summarization = st.selectbox('What method would you like to use for summarization?', ('Word Count Frequency', 'TextRank Algorithm'), index=None,
                                                  placeholder="Select summarization method", )
            if selected_summarization =="Word Count Frequency":
                if sentence_number > 0:
                    summary, word_freq = word_freq_summarize(text, sentence_number)
                    # st.write(word_freq)
                    st.write("WordCloud of the given context:")
                    plot_wordcloud(word_freq)
                    st.write(f"**Summarized Text:** {summary}")
                else:
                    st.write("Please enter a valid number for sentences.")

            elif selected_summarization =="TextRank Algorithm":
                if sentence_number > 0:
                    result = textrank_summarize(text, sentence_number)
                    st.write(f"**Summarized Text:** {result}")
                else:
                    st.write("Please enter a valid number for sentences.")
        else:
            st.write("Please enter text for summarization.")
    else:
        st.write("Please enter valid input")

if __name__ == '__main__':
    main()
    # test_wordcloud()
