import sys
from llama_cpp import llama_backend_init, llama_print_system_info

def check_gpu():
    print("--- Llama-cpp-python System Info ---", flush=True)
    try:
        # 顯示系統資訊，包含是否啟用 CUDA/Vulkan 等
        info = llama_print_system_info()
        # llama_print_system_info() 通常直接印到 stderr/stdout 並回傳 None
        # 下面這行是為了保證我們有抓到一些字串來顯示
        import llama_cpp
        print(f"llama-cpp-python version: {llama_cpp.__version__}", flush=True)
    except Exception as e:
        print(f"Error checking system info: {e}", flush=True)

if __name__ == "__main__":
    check_gpu()
