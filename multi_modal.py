from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.core.node_parser import SentenceSplitter
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import streamlit as st
import os
import requests
from pathlib import Path
import urllib.request
import torch
import clip
from PIL import Image
from llama_index.core.schema import ImageNode
from llama_index.core.response.notebook_utils import (
    display_source_node,
    display_image_uris,
)

os.system("cls")
# embed_model = ClipEmbedding(model_name="ViT-B/32")
device = "cuda" if torch.cuda.is_available() else "cpu"
# embed_model, preprocess = clip.load("ViT-B/32", device=device)
encode_kwargs = {'batch_size': 16}
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Settings.embed_model = embed_model

def get_wikipedia_images(title):
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|dimensions|mime",
            "generator": "images",
            "gimlimit": "50",
        },
    ).json()
    image_urls = []
    for page in response["query"]["pages"].values():
        if page["imageinfo"][0]["url"].endswith(".jpg") or page["imageinfo"][
            0
        ]["url"].endswith(".png"):
            image_urls.append(page["imageinfo"][0]["url"])
    return image_urls


image_uuid = 0
MAX_IMAGES_PER_WIKI = 20

wiki_titles = {
    "Tesla Model X",
    "Pablo Picasso",
    "Rivian",
    "The Lord of the Rings",
    "The Matrix",
    "The Simpsons",
}

data_path = Path("mixed_wiki")
if not data_path.exists():
    Path.mkdir(data_path)

# for title in wiki_titles:
#     response = requests.get(
#         "https://en.wikipedia.org/w/api.php",
#         params={
#             "action": "query",
#             "format": "json",
#             "titles": title,
#             "prop": "extracts",
#             "explaintext": True,
#         },
#     ).json()
#     page = next(iter(response["query"]["pages"].values()))
#     wiki_text = page["extract"]

#     with open(data_path / f"{title}.txt", "w", encoding="utf-8") as fp:
#         fp.write(wiki_text)

#     images_per_wiki = 0
#     try:
#         # page_py = wikipedia.page(title)
#         list_img_urls = get_wikipedia_images(title)
#         # print(list_img_urls)

#         for url in list_img_urls:
#             if url.endswith(".jpg") or url.endswith(".png"):
#                 image_uuid += 1
#                 # image_file_name = title + "_" + url.split("/")[-1]

#                 urllib.request.urlretrieve(
#                     url, data_path / f"{image_uuid}.jpg"
#                 )
#                 images_per_wiki += 1
#                 # Limit the number of images downloaded per wiki page to 15
#                 if images_per_wiki > MAX_IMAGES_PER_WIKI:
#                     break
#     except:
#         print(str(Exception("No images found for Wikipedia page: ")) + title)
#         continue

image_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()

# create client and a new collection
chroma_client = chromadb.PersistentClient()
chroma_collection = chroma_client.get_or_create_collection(
    "multimodal_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

# load documents
documents = SimpleDirectoryReader("./mixed_wiki/").load_data()
# Create transformation
text_splitter = SentenceSplitter(chunk_size=256)
# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context = storage_context,
    # embed_model = embed_model,
    transformations=[text_splitter]
)

retriever = index.as_retriever(similarity_top_k=50)
retrieval_results = retriever.retrieve("Picasso famous paintings")


image_results = []
MAX_RES = 5
cnt = 0
for r in retrieval_results:
    if isinstance(r.node, ImageNode):
        image_results.append(r.node.metadata["file_path"])
    else:
        if cnt < MAX_RES:
            display_source_node(r)
        cnt += 1

display_image_uris(image_results, [3, 3], top_k=2)