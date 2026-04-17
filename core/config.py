import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class PipelineConfig:
    embed_model_repo: str
    embed_model_file: str
    embed_n_ctx: int
    summary_model_repo: str
    summary_model_file: str
    summary_temp: float
    summary_top_p: float
    summary_top_k: int
    n_ctx: int
    n_gpu_layers: int

class ConfigService:
    """
    配置服務 (.NET Style)
    職責：專門負責載入與讀取環境變數，是專案中唯一存取環境變數的出口
    """
    def __init__(self):
        # 載入 .env
        load_dotenv()
        
    def get_config(self) -> PipelineConfig:
        """
        從環境變數組裝 PipelineConfig 物件
        """
        return PipelineConfig(
            embed_model_repo=os.getenv("EMBED_MODEL_REPO", "phate334/multilingual-e5-large-gguf"),
            embed_model_file=os.getenv("EMBED_MODEL_FILE", "multilingual-e5-large-f16.gguf"),
            embed_n_ctx=int(os.getenv("EMBED_N_CTX", 512)),
            summary_model_repo=os.getenv("SUMMARY_MODEL_REPO", "unsloth/gemma-4-E4B-it-GGUF"),
            summary_model_file=os.getenv("SUMMARY_MODEL_FILE", "gemma-4-E4B-it-Q4_K_M.gguf"),
            summary_temp=float(os.getenv("SUMMARY_TEMP", 1.0)),
            summary_top_p=float(os.getenv("SUMMARY_TOP_P", 0.95)),
            summary_top_k=int(os.getenv("SUMMARY_TOP_K", 64)),
            n_ctx=int(os.getenv("N_CTX", 32768)),
            n_gpu_layers=int(os.getenv("N_GPU_LAYERS", -1))
        )
