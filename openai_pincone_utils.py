import tiktoken, configparser, openai, pinecone, os
from openai.error import RateLimitError, InvalidRequestError, APIError
import time
import pandas as pd
from tqdm.auto import tqdm
import sys
from pinecone import Pinecone, ServerlessSpec

import asyncio


encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
openai.api_key = os.getenv("OPENAI_API_KEY")
SMART_CHAT_MODEL = "gpt-4"
FAST_CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
PINCONE_API_KEY = os.getenv("PINCONE_API_KEY")
PINCONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
PINCONE_ENV = os.getenv("PINCONE_ENV")


def count_tokens(text):
    tokens = len(encoding.encode(text))
    return tokens


def get_embedding(text: str, model: str = EMBEDDING_MODEL):
    while True:
        try:
            result = openai.Embedding.create(model=model, input=text)
            break
        except (APIError, RateLimitError):
            print("OpenAI had an issue, trying again in a few seconds...")
            time.sleep(10)
    return result["data"][0]["embedding"]


def create_embeddings_dataframe(context_chunks):
    embeddings = []
    progress_bar = tqdm(
        total=len(context_chunks), desc="Calculating embeddings", position=0
    )

    for chunk in context_chunks:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
        progress_bar.update(
            1
        )  # Increment the progress bar after each embedding calculation
        sys.stdout.flush()

    progress_bar.close()  # Close the progress bar when the loop is finished

    df = pd.DataFrame({"index": range(len(context_chunks)), "chunk": context_chunks})

    embeddings_df = pd.DataFrame(
        embeddings, columns=[f"embedding{i}" for i in range(1536)]
    )

    result_df = pd.concat([df, embeddings_df], axis=1)

    return result_df


def store_embeddings_in_pinecone(
    index=PINCONE_INDEX_NAME,
    pinecone_api_key=PINCONE_API_KEY,
    pinecone_env=PINCONE_ENV,
    data_frame=None,
):
    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    pinecone_index = pc.Index(name=index)

    if data_frame is not None and not data_frame.empty:
        batch_size = 80
        vectors_to_upsert = []
        batch_count = 0
        total_batches = -(-len(data_frame) // batch_size)

        progress_bar = tqdm(
            total=total_batches, desc="Loading info into Pinecone", position=0
        )

        for index, row in data_frame.iterrows():
            context_chunk = row["chunk"]

            vector = [float(row[f"embedding{i}"]) for i in range(1536)]

            pine_index = f"hw_{index}"
            metadata = {"context": context_chunk}
            vectors_to_upsert.append((pine_index, vector, metadata))

            if len(vectors_to_upsert) == batch_size or index == len(data_frame) - 1:
                while True:
                    try:
                        upsert_response = pinecone_index.upsert(
                            vectors=vectors_to_upsert
                        )

                        batch_count += 1
                        vectors_to_upsert = []

                        progress_bar.update(1)
                        sys.stdout.flush()
                        break

                    except pinecone.core.client.exceptions.ApiException:
                        print(
                            "Pinecone is a little overwhelmed, trying again in a few seconds..."
                        )
                        time.sleep(10)

        progress_bar.close()

    else:
        print("No dataframe to retrieve embeddings")
