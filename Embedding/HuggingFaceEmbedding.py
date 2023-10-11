import requests
from retry import retry
import numpy as np
import os


#read from env the hf token
if os.environ.get("HUGGINGFACEHUB_API_TOKEN") is not None:
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
else:
    raise Exception("You must provide the huggingface token")
  
# model_id = "sentence-transformers/all-MiniLM-L6-v2" NOT WORKING FROM 10/05/2023
model_id = "obrizum/all-MiniLM-L6-v2"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}


def reshape_array(arr):
    # create an array of zeros with shape (1536)
    new_arr = np.zeros((1536,))
    # copy the original array into the new array
    new_arr[:arr.shape[0]] = arr
    # return the new array
    return new_arr

@retry(tries=3, delay=10)
def newEmbeddings(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    result = response.json()
    if isinstance(result, list):
      return result
    elif list(result.keys())[0] == "error":
      raise RuntimeError(
          "The model is currently loading, please re-run the query."
          )
      
def newEmbeddingFunction(texts):
    embeddings = newEmbeddings(texts)
    embeddings = np.array(embeddings, dtype=np.float32)
    shaped_embeddings = reshape_array(embeddings)
    return shaped_embeddings

def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        import sentence_transformers

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if self.multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(texts, **self.encode_kwargs)

        return embeddings.tolist()