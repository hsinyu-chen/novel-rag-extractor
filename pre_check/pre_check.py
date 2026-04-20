import os
import time
import sys
from dotenv import load_dotenv
from openai import OpenAI

def check_weaviate():
    """測試 Weaviate 連線"""
    try:
        import weaviate
        from weaviate.connect import ConnectionParams
        
        host = os.getenv("WEAVIATE_HOST", "localhost")
        h_port = int(os.getenv("WEAVIATE_HTTP_PORT", 8080))
        h_secure = os.getenv("WEAVIATE_HTTP_SECURE", "False").lower() == "true"
        g_port = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
        g_secure = os.getenv("WEAVIATE_GRPC_SECURE", "False").lower() == "true"

        client = weaviate.WeaviateClient(
            connection_params=ConnectionParams.from_params(
                http_host=host,
                http_port=h_port,
                http_secure=h_secure,
                grpc_host=host,
                grpc_port=g_port,
                grpc_secure=g_secure,
            )
        )
        try:
            client.connect()
            return "true" if client.is_ready() else "Exception: Weaviate not ready"
        finally:
            client.close()
    except Exception as e:
        return f"Exception: {str(e)}"

def check_all_servers():
    """整合測試：檢查 LLM Server 與 Embedding Server 是否正常回應"""
    try:
        llm_url = os.getenv("SUMMARY_BASE_URL", "http://127.0.0.1:8080/v1")
        llm_key = os.getenv("SUMMARY_API_KEY", "no-key")
        llm_model = os.getenv("SUMMARY_MODEL", "any")
        
        emb_url = os.getenv("EMBED_BASE_URL", "http://127.0.0.1:8081/v1")
        emb_key = os.getenv("EMBED_API_KEY", "no-key")
        emb_model = os.getenv("EMBED_MODEL", "any")
        
        print(f"Checking Servers: [LLM: {llm_url}] & [Embed: {emb_url}]...", flush=True)
        
        # 1. 測試 Embedding
        emb_client = OpenAI(base_url=emb_url, api_key=emb_key)
        emb_res = emb_client.embeddings.create(input="Testing embedding server.", model=emb_model)
        emb_status = f"OK (Dim: {len(emb_res.data[0].embedding)})" if emb_res else "Failed"
        
        # 2. 測試 LLM (Gemma-4)
        llm_client = OpenAI(base_url=llm_url, api_key=llm_key)
        prompt = "Hello! Who are you? Reply in one short sentence."
        start_time = time.time()
        llm_res = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0.3
        )
        duration = time.time() - start_time
        
        if llm_res and llm_res.choices:
            text = llm_res.choices[0].message.content.strip()
            # 粗估 TPS
            tokens = llm_res.usage.completion_tokens
            tps = tokens / duration if duration > 0 else 0
            
            print(f"--- Server Status Report ---")
            print(f"Embedding_Server: {emb_status}")
            print(f"LLM_Server_Speed: {tps:.2f} t/s")
            print(f"LLM_Server_Response: {text}")
            return "true"
        return "Exception: Server tests failed"
    except Exception as e:
        return f"Exception: {str(e)}"

def run_pre_requirements() -> bool:
    results = {
        "Weaviate_Connectivity": check_weaviate,
        "Llama_Server_Stack_Test": check_all_servers
    }
    
    overall_success = True
    print("\n--- SYSTEM PRE-CHECK ---", flush=True)
    for key, func in results.items():
        res = func()
        if res != "true":
            print(f"{key}: {res}", flush=True)
            overall_success = False
        else:
            if key == "Weaviate_Connectivity": # 也稍微提示一下 Weaviate OK
                 print(f"{key}: OK", flush=True)

    print("-" * 30 + "\n", flush=True)
    return overall_success

if __name__ == "__main__":
    if not run_pre_requirements():
        sys.exit(1)
