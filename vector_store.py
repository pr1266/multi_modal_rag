# import qdrant_client
# from llama_index.core import SimpleDirectoryReader
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# from llama_index.core import VectorStoreIndex, StorageContext
# from llama_index.core.indices import MultiModalVectorStoreIndex

# Create a local Qdrant vector store
# client = qdrant_client.QdrantClient(path="qdrant_d_0")

# text_store = QdrantVectorStore(
#     client=client, collection_name="text_collection_0"
# )
# image_store = QdrantVectorStore(
#     client=client, collection_name="image_collection_0"
# )
# storage_context = StorageContext.from_defaults(
#     vector_store=text_store, image_store=image_store
# )

# # Create the MultiModal index
# documents = SimpleDirectoryReader("./data_wiki/").load_data()
# index = MultiModalVectorStoreIndex.from_documents(
#     documents,
#     storage_context=storage_context,
# )

from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import numpy as np
import os
from tqdm import tqdm

os.system("cls")

db_path = './vector_db'
data_path = "./data_wiki/"
image_path = data_path
image_loader = ImageLoader()


client = chromadb.PersistentClient(path = db_path)
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

collection = client.create_collection(
    name = "pr_multimodal",
    embedding_function = embedding_function,
    data_loader = data_loader
)


def add_images_to_collection(folder_path):
    image_files = [os.path.join(folder_path, image_name) for image_name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, image_name)) and image_name.lower().endswith((".png", ".jpg"))]

    for image_path in tqdm(image_files, desc = "creating image embeddings and adding to db"):
        try:
            image = np.array(Image.open(image_path))
            collection.add(
                ids = [os.path.basename(image_path)],
                images = [image]
            )
        except Exception as e:
            print(f"Error Processing {image_path}: {e}")

add_images_to_collection(image_path)