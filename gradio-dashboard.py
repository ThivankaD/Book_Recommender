import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader
import re
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"]=books["thumbnail"] + "&fife=w800"
books["large_thumbnail"]=np.where(
    books["large_thumbnail"].isna(),
    "https://i.pinimg.com/736x/ac/6e/2b/ac6e2bc17022069b8f424c162382f021.jpg",
    books["large_thumbnail"]
)

raw_documents= TextLoader("tagged_description.txt",encoding="utf-8").load()
text = raw_documents[0].page_content
matches = re.findall(r'(978\d{10})\s+(.*?)(?=(978\d{10})|$)', text, flags=re.DOTALL)
documents = [
    Document(page_content=desc.strip(), metadata={"isbn": isbn})
    for isbn, desc, _ in matches
]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma(
    persist_directory="./book_vectors",  # path to your folder
    embedding_function=embeddings
)


def retrieve_semantic_recommendations(
        query:str,
        category: str = None,
        tone:str = None,
        initial_top_k: int=50,
        final_top_k:int=16,

)->pd.DataFrame:
    recs =db_books.similarity_search(query,k=initial_top_k)
    books_list = []
    for doc in recs:
        if "isbn" in doc.metadata:
            try:
                books_list.append(int(doc.metadata["isbn"]))
            except ValueError:
                # skip if ISBN is not a valid number
                continue
    book_recs=books[books["isbn13"].isin(books_list)].head(final_top_k)


    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"]==category][:final_top_k]

    else:
        books_recs =book_recs.head(final_top_k)


    if tone == "Happy":
        book_recs.sort_values(by="joy",ascending=False,inplace=True)

    elif tone == "Surprising":
        book_recs.sort_values(by="surprise",ascending=False,inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger",ascending=False,inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear",ascending=False,inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness",ascending=False,inplace=True)
    return book_recs


def recommend_books(
        query:str,
        category: str,
        tone:str
):
    recommendations = retrieve_semantic_recommendations(query,category,tone)
    results=[]
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split=description.split()
        truncated_description=" ".join(truncated_desc_split[:30]) + "..."

        authors_split=row["authors"].split(";")
        if len(authors_split)==2:
            authors_str= f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split)>2:
            authors_str=f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str=row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append([row["large_thumbnail"], caption])

    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"]+["Happy","Suprising","Angry","Suspenseful","Sad"]

with gr.Blocks(theme =gr.themes.Glass()) as dashboard:
    gr.Markdown("## Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please Enter Your Book Description",placeholder="e.g.., A story about forgiveness...")
        category_dropdown = gr.Dropdown(choices = categories,label="Select Category",value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an Emotional tone", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommendations", columns=8,rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query,category_dropdown,tone_dropdown],
                        outputs=[output])

    if __name__ == "__main__":
        dashboard.launch()


