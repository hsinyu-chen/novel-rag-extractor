from dependency_injector import containers, providers
from huggingface_hub import hf_hub_download

from processor.json_storage import JsonStorage
from processor.llm_engine import NativeLlamaEngine
from processor.scene_validator import SceneValidator
from processor.scene_summarizer import SceneSummarizer
from modules.book_pre_process import BookPreProcessor

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
        model_path=providers.Callable(
            hf_hub_download,
            repo_id=config.summary_model_repo,
            filename=config.summary_model_file
        ),
        params=providers.Dict(
            temperature=config.summary_temp.as_float(),
            top_p=config.summary_top_p.as_float(),
            top_k=config.summary_top_k.as_int(),
            n_ctx=config.n_ctx.as_int(),
            n_gpu_layers=config.n_gpu_layers.as_int(),
        )
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

    # 註冊預處理器
    pre_processor = providers.Singleton(
        BookPreProcessor,
        storage=storage,
        validator=validator,
        summarizer=summarizer,
        config=config
    )
