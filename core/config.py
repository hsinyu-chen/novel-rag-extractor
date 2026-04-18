import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class PipelineConfig:
    embed_base_url: str
    embed_api_key: str
    embed_model: str
    summary_base_url: str
    summary_api_key: str
    summary_model: str
    summary_temp: float
    summary_top_p: float
    summary_top_k: int
    weaviate_host: str
    weaviate_http_port: int
    weaviate_http_secure: bool
    weaviate_grpc_port: int
    weaviate_grpc_secure: bool

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
            embed_base_url=os.getenv("EMBED_BASE_URL", "http://127.0.0.1:8081/v1"),
            embed_api_key=os.getenv("EMBED_API_KEY", "no-key-required"),
            embed_model=os.getenv("EMBED_MODEL", "multilingual-e5-large-f16"),
            
            summary_base_url=os.getenv("SUMMARY_BASE_URL", "http://127.0.0.1:8080/v1"),
            summary_api_key=os.getenv("SUMMARY_API_KEY", "no-key-required"),
            summary_model=os.getenv("SUMMARY_MODEL", "gemma-4-E4B-it-Q4_K_M"),
            
            summary_temp=float(os.getenv("SUMMARY_TEMP", 1.0)),
            summary_top_p=float(os.getenv("SUMMARY_TOP_P", 0.95)),
            summary_top_k=int(os.getenv("SUMMARY_TOP_K", 64)),
            
            weaviate_host=os.getenv("WEAVIATE_HOST", "localhost"),
            weaviate_http_port=int(os.getenv("WEAVIATE_HTTP_PORT", 8080)),
            weaviate_http_secure=os.getenv("WEAVIATE_HTTP_SECURE", "False").lower() == "true",
            weaviate_grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", 50051)),
            weaviate_grpc_secure=os.getenv("WEAVIATE_GRPC_SECURE", "False").lower() == "true"
        )
