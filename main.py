import os
import argparse
from dataclasses import asdict
from core.config import ConfigService
from core.container import AppContainer
from pre_check import run_pre_requirements

def main():
    """
    程式進入點
    流程：Pre-check -> 組建配置 -> 註冊服務 -> 解析對象 -> 執行業務
    """
    # 0. 執行系統 Pre-check (失敗則退出)
    if not run_pre_requirements():
        print("[Error] System pre-check failed. Please check your servers and .env configuration.")
        return
        
    # 1. 解析命令列參數
    parser = argparse.ArgumentParser(description="Narrative RAG Pipeline Entry Point")
    parser.add_argument("--novel", type=str, required=True, help="小說資料夾名稱")
    parser.add_argument("--start", type=int, default=1, help="起始集數數字")
    parser.add_argument("--vol", type=int, default=0, help="只跑指定的單一集數")
    parser.add_argument("--clean", action="store_true", help="清除該小說的全部輸出後重跑")
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
        start = args.vol if args.vol else args.start
        end = args.vol if args.vol else 0
        pre_processor.run(args.novel, start, end_vol=end, clean_output=args.clean)
    
    if args.mode in ["process", "all"]:
        print(f"\n[Mode: Process] Starting knowledge extraction agent for {args.novel}...")
        knowledge_processor = container.knowledge_processor()
        start = args.vol if args.vol else args.start
        end = args.vol if args.vol else 0
        knowledge_processor.run(args.novel, start, end_vol=end, clean_output=args.clean)

if __name__ == "__main__":
    main()
