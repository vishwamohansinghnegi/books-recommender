import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr
load_dotenv()

df = pd.read_csv('books_sentiment.csv')

# For showing books cover clearly
df['large_thumbnail'] = df['thumbnail'] + '&fife=w800'

df['large_thumbnail'] = np.where(df['large_thumbnail'].isnull(), 'Cover Not Found!!', df['large_thumbnail'])

raw_docs = TextLoader('tagged_description.txt', encoding='utf-8')
raw_docs = raw_docs.load()
text_splitter = RecursiveCharacterTextSplitter(separators=['\n'], chunk_size=50, chunk_overlap=0)
docs = text_splitter.split_documents(raw_docs)

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

books_db = Chroma.from_documents(
    docs,
    embedding = embedding_model
)

def retrieve_recommendations(query, category, tone, top_k: int=14):
    retr= books_db.similarity_search(query, k=top_k)
    books_isbn = [int(rec.page_content.strip('\n"').split()[0]) for rec in retr]
    books_recm = df[df['isbn13'].isin(books_isbn)]

    if category!='All':
        books_recm = books_recm[books_recm['simple_categories'] == category]
    

    if tone=='Happy':
        books_recm.sort_values(by='joy', ascending=False, inplace=True)

    elif tone=='Surprising':
        books_recm.sort_values(by='surprise', ascending=False, inplace=True)
    
    elif tone=='Angry':
        books_recm.sort_values(by='angry', ascending=False, inplace=True)

    elif tone=='Sad':
        books_recm.sort_values(by='sad', ascending=False, inplace=True)
    
    elif tone=='Thriller/Fearful':
        books_recm.sort_values(by='fear', ascending=False, inplace=True)
    return books_recm


def recommend_books(query, category, tone):

    books_recm = retrieve_recommendations(query, category, tone)
    results = []

    for _, row in books_recm.iterrows():

        description = row['description']
        truncated_description = ' '.join(description.split()[:30]) + '...'    # For showing in gradio

        authors_split = row['authors'].split(';')

        if len(authors_split)==1:
            author_str = row['authors']
        elif len(authors_split)==2:
            author_str = f'{authors_split[0]} and {authors_split[1]}'
        else:
            author_str = ', '.join(authors_split)[:-1] + 'and ' + authors_split[-1]

        caption = f'{row['title']} by {author_str}: {truncated_description}'
        results.append((row['large_thumbnail'], caption))
    
    return results

categories = ["All"] + sorted(df["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Thriller/Fearful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about warriors and kings!")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 7, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()