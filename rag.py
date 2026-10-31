import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_classic.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from prompt import COMBINE_PROMPT, EXAMPLE_PROMPT, QUESTION_PROMPT

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        client = QdrantClient(
            url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"]
        )
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE
                )
            )
        collection_info = client.get_collection(COLLECTION_NAME)
        if "metadata.source" not in collection_info.payload_schema:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        if "ingestion_complete" not in collection_info.payload_schema:
            client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="ingestion_complete",
        field_schema=models.PayloadSchemaType.BOOL,
        wait=True
    )        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=ef
        )    
def url_exists(url, client):


    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source", match=models.MatchValue(value=url)
                ),
                models.FieldCondition(
                    key="ingestion_complete", match=models.MatchValue(value=True)
                ),
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )

    return len(points) > 0
def process_urls(urls):

    yield "Initializing components..."
    initialize_components()

    client = vector_store.client

    for url in dict.fromkeys(url.strip() for url in urls if url.strip()):
        yield f"Checking URL: {url}"

        # Skip articles that have already been processed
        if url_exists(url, client):
            yield f"Already processed. Reusing embeddings: {url}"
            continue

        yield f"Loading article: {url}"

        loader = UnstructuredURLLoader(urls=[url])
        data = loader.load()

        if not data or not any(doc.page_content.strip() for doc in data):
            raise RuntimeError(f"No usable content retrieved from: {url}")

        # Split article into chunks
        yield "Splitting article into chunks..."

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=200,
        )

        docs = text_splitter.split_documents(data)

        if not docs:
            raise RuntimeError(f"No chunks generated for: {url}")

        # Ensure each chunk has the correct source URL
        for doc in docs:
            doc.metadata["source"] = url

        # Remove any chunks left by a previous incomplete upload
        source_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source", match=models.MatchValue(value=url)
                )
            ]
        )

        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(filter=source_filter),
            wait=True,
        )

        # Generate unique IDs for the new chunks
        uuids = [str(uuid4()) for _ in docs]

        # Store documents and generate embeddings
        yield f"Generating and storing embeddings: {url}"

        vector_store.add_documents(documents=docs, ids=uuids)

        # Mark the upload complete only after all chunks are stored
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"ingestion_complete": True},
            points=uuids,
            wait=True,
        )

        yield f"Successfully processed: {url}"

    yield "All URLs processed successfully!"


def generate_answer(query, urls):

    initialize_components()

    if not urls:
        raise ValueError("At least one URL is required.")

    url_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.source", match=models.MatchAny(any=urls)
            )
        ]
    )

    retriever = vector_store.as_retriever(search_kwargs={"filter": url_filter, "k": 4})

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=retriever,
        question_prompt=QUESTION_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        document_prompt=EXAMPLE_PROMPT,
    )

    result = chain.invoke({"question": query}, return_only_outputs=True)

    sources = result.get("sources", "")

    return result["answer"], sources
