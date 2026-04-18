import requests
from openai import OpenAI

class LlamaSimpleEmbeddings:
    """
    通用 Embedding Wrapper，提供 LangChain 或是自訂的 embed 介面。
    已經修正為 multilingual-e5 要求的 passage/query 前綴。
    支援 llama-server 原生的 /tokenize 端點。
    """
    def __init__(self, base_url: str, api_key: str, model: str):
        print(f"Connecting to Embedding API at {base_url} (model: {model})...")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        # 取得不含 /v1 的 base hostname:port 供 tokenize 使用 (llama-server 專有功能)
        self.tokenize_url = base_url.replace("/v1", "") + "/tokenize"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # indexing
        formatted_texts = [f"passage: {t}" for t in texts]
        res = self.client.embeddings.create(input=formatted_texts, model=self.model)
        return [r.embedding for r in res.data]

    def embed_query(self, text: str) -> list[float]:
        # searching
        res = self.client.embeddings.create(input=f"query: {text}", model=self.model)
        return res.data[0].embedding
        
    def tokenize(self, text: str) -> int:
        try:
            res = requests.post(self.tokenize_url, json={"content": text})
            if res.status_code == 200:
                return len(res.json().get("tokens", []))
            return 0
        except Exception:
            return 0
