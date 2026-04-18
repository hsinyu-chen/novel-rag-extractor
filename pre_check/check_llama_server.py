import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# 載入 .env
load_dotenv()

def test_llama_server():
    # 從 .env 讀取參數
    base_url = os.getenv("SUMMARY_BASE_URL", "http://127.0.0.1:8080/v1")
    model = os.getenv("SUMMARY_MODEL", "gemma-4-E4B-it-Q4_K_M")
    
    print(f"--- 測試伺服器: {base_url} (Model: {model}) ---", flush=True)
    
    client = OpenAI(base_url=base_url, api_key="no-key")

    print("\n--- 正在測試推理 ---", flush=True)
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "介紹你自己"}],
            max_tokens=128,
            temperature=0.3
        )
        end_time = time.time()
        
        text = response.choices[0].message.content.strip()
        usage = response.usage
        duration = end_time - start_time
        
        print("\n" + "="*40, flush=True)
        print("AI: " + text, flush=True)
        print("-" * 40, flush=True)
        print(f"Speed: {usage.completion_tokens/duration:.2f} tokens/sec", flush=True)
        print("="*40, flush=True)
    except Exception as e:
        print(f"推理失敗: {e}", flush=True)

if __name__ == "__main__":
    test_llama_server()
