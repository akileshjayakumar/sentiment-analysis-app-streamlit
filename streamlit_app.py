import streamlit as st
from langchain.llms import OpenAI
import matplotlib.pyplot as plt
import os


api_key = os.getenv("LANGCHAIN_API_KEY")
llm = OpenAI(api_key=api_key)


# Page configuration
st.set_page_config(page_title='Sentiment Analysis App', page_icon='📊')
st.title('📊 Sentiment Analysis App')


def clean_text(text):
    """Basic text cleaning to remove unnecessary characters and normalize text."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text.strip()


def get_sentiment(comment):
    """Perform sentiment analysis using LangChain."""
    cleaned_comment = clean_text(comment)
    prompt = f"""Analyze the sentiment of the following comment and classify it as Positive, Neutral,
                or Negative with a detailed explanation: "{cleaned_comment}"""
    try:
        response = llm.generate([prompt], max_tokens=150)
        sentiment_result = response.generations[0][0].text.strip()
        return sentiment_result
    except Exception as e:
        return f"Error occurred: {str(e)}"



def main():
    # Custom CSS for improved UI
    st.markdown("""
        <style>
        .reportview-container {
            padding: 5rem 1rem;  # Top and Bottom Padding, Left and Right Padding
        }
        .big-font {
            font-family: 'Helvetica', sans-serif; 
            font-size:20px !important;
            font-weight: bold;
        }
        .stTextArea>textarea {
            height: 300px;
            font-family: 'Helvetica', sans-serif;
        }
        .stButton>button {
            width: 100px;
            border-radius: 20px;
            background-color: #0c008c;
            color: white;
            font-size: 16px;
            height: 2.5em;  # Button height
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("##")
    st.markdown("""
        <div style='text-align: left'>
            Enter the comments below, separated by new lines, and press 'Analyse' to determine their sentiments.
        </div>
        """, unsafe_allow_html=True)

    comments = st.text_area("", value="", height=300,
                            max_chars=1500, key="comments")

    if st.button("Analyse", key="analyze"):
        with st.spinner('Analyzing sentiments...'):
            if comments:
                results = []
                sentiments = []
                # To create a line graph
                sentiment_order = {'Positive': 2,
                                   'Neutral': 1, 'Negative': 0, 'Error': -1}
                for comment in comments.split('\n'):
                    if comment.strip():
                        result = get_sentiment(comment)
                        results.append({'comment': comment, 'result': result})
                        sentiment = result.split('.')[0]
                        sentiments.append(sentiment_order.get(sentiment, -1))

                if results:
                    st.success("Analysis completed!")
                    for res in results:
                        st.markdown(f"#### Comment:")
                        st.write(f"{res['comment']}")
                        st.markdown(f"#### Analysis:")
                        st.write(f"{res['result']}")

                # Visualization
                if sentiments:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(sentiments, marker='o', linestyle='-', color='b')
                    ax.set_title("Sentiment Trend Over Comments")
                    ax.set_xlabel("Comment Index")
                    ax.set_ylabel("Sentiment Score")
                    ax.set_yticks([2, 1, 0, -1])
                    ax.set_yticklabels(
                        ['Positive', 'Neutral', 'Negative', 'Error'])
                    st.pyplot(fig)
            else:
                st.warning("Please enter at least one comment for analysis.")


if __name__ == "__main__":
    main()

# Footer
st.markdown(
    """
    <footer class="footer">
        <small>
            © Akilesh Jayakumar | Built with Python, LangChain and Streamlit
        </small>
    </footer>
    """, unsafe_allow_html=True
)
