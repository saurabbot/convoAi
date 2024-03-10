import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import openai, tiktoken
from pinecone import Pinecone


openai.api_key = os.getenv("OPENAI_API_KEY")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
SMART_CHAT_MODEL = "gpt-4"
FAST_CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
PINCONE_API_KEY = os.getenv("PINCONE_API_KEY")
PINCONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
PINCONE_ENV = os.getenv("PINCONE_ENV")


def seed_db():
    print("Check if index exists")
    pc = Pinecone(api_key=PINCONE_API_KEY, environment=PINCONE_ENV)
    pinecone_index = pc.Index(name=PINCONE_INDEX_NAME)

    loader = UnstructuredPDFLoader(
        "How_Conversational_Business_Can_Help_You_Get_and_Stay_Closer_to_Customers.pdf"
    )
    data = loader.load()
    print(data)
    print(f"You have loaded a PDF with {len(data)} pages")
    print(f"There are {len(data[0].page_content)} characters in your document")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(data)

    print(f"You have split your document into {len(texts)} smaller documents")
    print("Creating embeddings and index...")
    embeddings = OpenAIEmbeddings(client="")
    docsearch = PineconeVectorStore.from_texts(
        [t.page_content for t in texts], embeddings, index_name=PINCONE_INDEX_NAME
    )

    print("Done!")


def query_db(string: str):
    print("Querying the database...")
    llm_chat = ChatOpenAI(
        temperature=0.9, max_tokens=150, model="gpt-3.5-turbo-0613", client=""
    )
    embeddings = OpenAIEmbeddings(client="")
    doc_search = PineconeVectorStore.from_existing_index(
        index_name=PINCONE_INDEX_NAME, embedding=embeddings
    )
    chain = load_qa_chain(llm_chat)
    search = doc_search.similarity_search(string)
    response = chain.run(input_documents=search, question=string)
    print("Response:", response)


query_db("what exactly is “conversational business”?")
