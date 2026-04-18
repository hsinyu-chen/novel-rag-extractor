import requests
from dependency_injector import containers, providers

from processor.json_storage import JsonStorage
from processor.weaviate_storage import WeaviateStorage
from processor.llm_engine import NativeLlamaEngine
from processor.embed_engine import LlamaSimpleEmbeddings
from processor.scene_validator import SceneValidator
from processor.scene_summarizer import SceneSummarizer
from processor.knowledge_agent import KnowledgeAgent
from modules.book_pre_process import BookPreProcessor
from modules.knowledge_process import KnowledgeProcess

class AppContainer(containers.DeclarativeContainer):
    """
    DI 容器 (.NET Style IServiceCollection)
    職責：統一管理物件的生命週期與依賴注入
    """
    
    # 定義配置
    config = providers.Configuration()

    # 註冊 LLM 引擎 (Singleton 以節省 VRAM)
    llm_engine = providers.Singleton(
        NativeLlamaEngine,
        base_url=config.summary_base_url,
        api_key=config.summary_api_key,
        model=config.summary_model,
        params=providers.Dict(
            temperature=config.summary_temp.as_float(),
            top_p=config.summary_top_p.as_float(),
            top_k=config.summary_top_k.as_int()
        )
    )

    # 註冊 Embedding Engine
    embed_engine = providers.Singleton(
        LlamaSimpleEmbeddings,
        base_url=config.embed_base_url,
        api_key=config.embed_api_key,
        model=config.embed_model
    )

    # 註冊場景驗證器
    validator = providers.Singleton(
        SceneValidator,
        engine=llm_engine
    )

    # 註冊場景摘要器
    summarizer = providers.Singleton(
        SceneSummarizer,
        engine=llm_engine
    )

    # 註冊 Storage
    storage = providers.Singleton(
        JsonStorage,
        base_dir="output"
    )

    weaviate_db = providers.Singleton(
        WeaviateStorage,
        config=config,
        embed_func=embed_engine
    )

    # 註冊預處理器
    pre_processor = providers.Singleton(
        BookPreProcessor,
        storage=storage,
        embed_engine=embed_engine,
        validator=validator,
        summarizer=summarizer,
        config=config
    )

    knowledge_agent = providers.Singleton(
        KnowledgeAgent,
        engine=llm_engine
    )

    knowledge_processor = providers.Singleton(
        KnowledgeProcess,
        storage=storage,
        weaviate_db=weaviate_db,
        agent=knowledge_agent,
        config=config
    )
