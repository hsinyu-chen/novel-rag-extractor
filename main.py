import os
import argparse
import llama_cpp
import ctypes

# 官方方式：攔截 C++ 層級的所有日誌
# 定義一個空的回調函數
@ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
def llama_log_callback(level, text, user_data):
    pass

# 在任何推理開始前註冊回調
llama_cpp.llama_log_set(llama_log_callback, None)

os.environ["GGML_PYTHON_LOG_LEVEL"] = "ERROR"

from dataclasses import asdict
from core.config import ConfigService
from core.container import AppContainer

def main():
    """
    程式進入點
    流程：組建配置 -> 註冊服務 -> 解析對象 -> 執行業務
    """
    # 1. 解析命令列參數
    parser = argparse.ArgumentParser(description="Narrative RAG Pipeline Entry Point")
    parser.add_argument("--novel", type=str, required=True, help="小說資料夾名稱")
    parser.add_argument("--start", type=int, default=1, help="起始集數數字")
    parser.add_argument("--mode", type=str, default="ingest", choices=["ingest", "process", "all"])
    args = parser.parse_args()

    # 2. 載入配置 (ConfigService)
    config_service = ConfigService()
    config = config_service.get_config()
    
    # 3. 初始化 DI 容器 (Container Setup)
    container = AppContainer()
    # 將型別安全物件轉換為字典填入容器配置
    container.config.from_dict(asdict(config))
    
    # 4. 根據模式執行業務 (Resolve and Run)
    if args.mode in ["ingest", "all"]:
        print(f"\n[Mode: Ingest] Starting pre-processing for {args.novel}...")
        # 從容器中解析出 pre_processor，所有依賴會自動注入
        pre_processor = container.pre_processor()
        pre_processor.run(args.novel, args.start)
    
    if args.mode in ["process", "all"]:
        print("\n[Mode: Process] Agent Analysis logic coming soon...")

if __name__ == "__main__":
    main()
