from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from secret_keys import HUGGINGFACE_TOKEN
# Set your Hugging Face Inference API key
api_key = HUGGINGFACE_TOKEN  # Replace with your actual API key

# Create an instance of HuggingFaceInferenceAPIEmbeddings
embedding_model = HuggingFaceInferenceAPIEmbeddings(api_key=api_key)

# Example usage: Embedding a query
query = "This is a test query."
query_embedding = embedding_model.embed_query(query)
print("Query Embedding:", query_embedding)

# Example usage: Embedding multiple documents
documents = ["Document 1", "Document 2", "Document 3"]
document_embeddings = embedding_model.embed_documents(documents)
print("Document Embeddings:", document_embeddings)