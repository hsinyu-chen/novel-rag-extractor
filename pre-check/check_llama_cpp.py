import os
import time
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# 載入 .env
load_dotenv()

def test_llama_cpp():
    # 從 .env 讀取參數
    model_repo = os.getenv("MODEL_REPO", "bartowski/gemma-2-9b-it-GGUF")
    model_file = os.getenv("MODEL_FILE", "gemma-2-9b-it-Q4_K_M.gguf")
    n_gpu = int(os.getenv("N_GPU_LAYERS", -1))
    n_ctx = int(os.getenv("N_CTX", 2048))
    
    print(f"--- 載入設定: {model_repo}/{model_file} (GPU Layers: {n_gpu}) ---", flush=True)
    
    try:
        model_path = hf_hub_download(repo_id=model_repo, filename=model_file)
        print(f"模型路徑: {model_path}", flush=True)
    except Exception as e:
        print(f"下載失敗: {e}", flush=True)
        return

    print("\n--- 正在載入模型到 GPU ---", flush=True)
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu, 
            n_ctx=n_ctx,
            verbose=True
        )
    except Exception as e:
        print(f"初始化失敗: {e}", flush=True)
        return

    prompt = "Q: 介紹你自己\nA:"
    print(f"\n--- 正在測試推理 ---", flush=True)
    
    start_time = time.time()
    output = llm(
        prompt,
        max_tokens=128,
        stop=["Q:", "\n"],
        echo=False
    )
    end_time = time.time()
    
    text = output["choices"][0]["text"].strip()
    usage = output["usage"]
    duration = end_time - start_time
    
    print("\n" + "="*40, flush=True)
    print("AI: " + text, flush=True)
    print("-" * 40, flush=True)
    print(f"Speed: {usage['completion_tokens']/duration:.2f} tokens/sec", flush=True)
    print("="*40, flush=True)

if __name__ == "__main__":
    test_llama_cpp()
