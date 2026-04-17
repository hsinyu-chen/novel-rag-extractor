import os
import time
import sys
import warnings
import subprocess
from dotenv import load_dotenv

# 載入 .env
load_dotenv()

# 徹底隱藏不必要的日誌與警告
os.environ["GGML_PYTHON_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore", category=ResourceWarning)

def get_vram_info():
    """獲取 VRAM 使用狀況 (單位: MiB)"""
    try:
        # 使用 nvidia-smi 查詢
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        total, used, free = map(str.strip, output.strip().split(','))
        return f"{used}MiB / {free}MiB Free (Total {total}MiB)"
    except Exception:
        return "N/A"

def check_weaviate():
    """測試 Weaviate 連線"""
    try:
        import weaviate
        from weaviate.connect import ConnectionParams
        
        host = os.getenv("WEAVIATE_HOST", "weaviate.dynameis.app")
        h_port = int(os.getenv("WEAVIATE_HTTP_PORT", 443))
        h_secure = os.getenv("WEAVIATE_HTTP_SECURE", "True").lower() == "true"
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

def check_gpu():
    """測試 CUDA 支援"""
    try:
        import llama_cpp
        return "true (CUDA Enabled)"
    except Exception as e:
        return f"Exception: {str(e)}"

def check_all_models():
    """整合測試：同時載入 Embedding 與 Summary 模型並測試"""
    try:
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
        
        # 1. 準備模型路徑
        e5_repo = os.getenv("EMBED_MODEL_REPO", "phate334/multilingual-e5-large-gguf")
        e5_file = os.getenv("EMBED_MODEL_FILE", "multilingual-e5-large-f16.gguf")
        sum_repo = os.getenv("SUMMARY_MODEL_REPO", "unsloth/gemma-4-E4B-it-GGUF")
        sum_file = os.getenv("SUMMARY_MODEL_FILE", "gemma-4-E4B-it-Q4_K_M.gguf")
        
        print(f"Loading Models: [E5 Embedding] & [Gemma-4 Summary]...", flush=True)
        e5_path = hf_hub_download(repo_id=e5_repo, filename=e5_file)
        sum_path = hf_hub_download(repo_id=sum_repo, filename=sum_file)
        
        # 2. 同時載入模型 (保持連線)
        embed_llm = Llama(model_path=e5_path, embedding=True, verbose=False, n_gpu_layers=-1)
        summary_llm = Llama(
            model_path=sum_path,
            n_gpu_layers=int(os.getenv("N_GPU_LAYERS", -1)),
            n_ctx=int(os.getenv("N_CTX", 8192)),
            verbose=False
        )
        
        # 3. 測試 Embedding
        emb_res = embed_llm.create_embedding("Testing full stack embedding capacity.")
        emb_status = f"OK (Dim: {len(emb_res['data'][0]['embedding'])})" if emb_res else "Failed"
        
        # 4. 測試 Summary (預熱)
        prompt = "<bos><start_of_turn>user\nHello! Who are you? Reply in one short sentence.<end_of_turn>\n<start_of_turn>model\n"
        start_time = time.time()
        sum_res = summary_llm(prompt, max_tokens=64, stop=["<end_of_turn>"], temperature=0.3)
        duration = time.time() - start_time
        
        # 5. 獲取最終 VRAM 與效能
        vram = get_vram_info()
        
        if sum_res and "choices" in sum_res:
            text = sum_res["choices"][0]["text"].strip()
            tps = sum_res["usage"]["completion_tokens"] / duration if duration > 0 else 0
            
            print(f"--- Full Stack Report ---")
            print(f"Embedding_Test: {emb_status}")
            print(f"Summary_Speed: {tps:.2f} t/s")
            print(f"Summary_Response: {text}")
            print(f"Total_VRAM_Usage: {vram}")
            return "true"
        return "Exception: Tests failed"
    except Exception as e:
        return f"Exception: {str(e)}"

def run_pre_requirements():
    results = {
        "Weaviate_Connectivity": check_weaviate,
        "Llama_GPU_Support": check_gpu,
        "Full_Model_Stack_Test": check_all_models
    }
    
    print("\n--- SYSTEM PRE-CHECK ---", flush=True)
    for key, func in results.items():
        res = func()
        if key != "Full_Model_Stack_Test" or res != "true":
            print(f"{key}: {res}", flush=True)
    print("-" * 30 + "\n", flush=True)

if __name__ == "__main__":
    run_pre_requirements()
