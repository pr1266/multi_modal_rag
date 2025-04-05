import clip
import torch
from PIL import Image
import wikipedia
import urllib.request
from pathlib import Path
import requests
import numpy as np
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os

os.system("cls")

db_path = "./vector_db"

client = chromadb.PersistentClient(path = db_path)
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

collection = client.get_or_create_collection(
    name = "pr_multimodal",
    embedding_function = embedding_function,
    data_loader = data_loader
)

st.title("Pr1266 Image Search Engine")

# Search bar
query = st.text_input("Enter your search query:")
parent_path = './data_wiki'
if st.button("Search"):
    results = collection.query(query_texts=[query], n_results=5,include=["distances"])
    print(results)
    for image_id, distance in zip(results['ids'][0], results['distances'][0]):
        image_path = os.path.join(parent_path, image_id)
        st.image(image_path, caption=os.path.basename(image_path))
        st.write(f"Distance: {distance}")