import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class Settings:
    # --- Directory ---
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.environ.get("CORPRORAG_DATA_DIR", root_dir + "/dickens/Musique")

    # --- Chunking ---
    chunk_token_size: int = int(os.environ.get("CHUNK_TOKEN_SIZE", "2000"))
    chunk_overlap_token_size: int = int(os.environ.get("CHUNK_OVERLAP_TOKEN_SIZE", "50"))

    # --- Model ---
    tiktoken_model_name: str = "gpt-4o-mini"
    llm_model_name: str = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini-2024-07-18")
    embedding_model_name: str = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    llm_model_max_token_size: int = 30000
    embedding_dim: int = 1024
    embedding_max_token_size: int = 8192

    # --- LLM Client ---
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "")
    llm_api_key: str = os.environ.get("LLM_API_KEY", "")
    llm_timeout: float = 460.0

    # Optional planner-only LLM for cor_simple decomposition. Leave empty to reuse llm_model_func.
    simple_planner_base_url: str = os.environ.get(
        "SIMPLE_PLANNER_BASE_URL", os.environ.get("PLANNER_BASE_URL", os.environ.get("OPENAI_BASE_URL", ""))
    )
    simple_planner_api_key: str = os.environ.get(
        "SIMPLE_PLANNER_API_KEY", os.environ.get("PLANNER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    )
    simple_planner_model_name: str = os.environ.get("SIMPLE_PLANNER_MODEL_NAME", os.environ.get("PLANNER_MODEL_NAME", ""))

    # --- Natural-language cor_simple ---
    cor_simple_max_correction_depth: int = int(os.environ.get("COR_SIMPLE_MAX_CORRECTION_DEPTH", "1"))
    cor_simple_ablation_variant: str = os.environ.get("COR_SIMPLE_ABLATION_VARIANT", "full")
    cor_simple_plan_replay_path: str = os.environ.get("COR_SIMPLE_PLAN_REPLAY_PATH", "")
    cor_simple_final_input_tokens: int = int(os.environ.get("COR_SIMPLE_FINAL_INPUT_TOKENS", "6200"))
    cor_simple_final_output_tokens: int = int(os.environ.get("COR_SIMPLE_FINAL_OUTPUT_TOKENS", "256"))

    cor_simple_enable_visual_verifier: bool = os.environ.get("COR_SIMPLE_VISUAL_VERIFIER", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    # --- LLM Parameters ---
    # Leave room for the input prompt on locally served models whose total
    # context window may be only 8K tokens.
    llm_max_tokens: Optional[int] = int(os.environ.get("LLM_MAX_TOKENS", "2048"))
    llm_temperature: Optional[float] = 0.7
    llm_top_p: Optional[float] = None
    llm_presence_penalty: Optional[float] = None
    llm_extra_body: Optional[Dict] = None

    # --- Embedding Client ---
    embedding_base_url: Optional[str] = os.environ.get("EMBEDDING_BASE_URL", "")
    embedding_api_key: Optional[str] = os.environ.get(
        "EMBEDDING_API_KEY", os.environ.get("EMBEDDING_BASE_KEY", "")
    )
    embedding_timeout: float = 120.0

    # --- Rerank Model ---
    rerank_base_url: str = os.environ.get("RERANK_BASE_URL", "")
    rerank_model_name: str = os.environ.get(
        "RERANK_MODEL_NAME", "Pro/BAAI/bge-reranker-v2-m3"
    )
    rerank_api_key: str = os.environ.get("RERANK_API_KEY", "")
    rerank_timeout: float = 60.0
    rerank_max_retries: int = 20
    rerank_max_chunks_per_doc: int = 8196
    rerank_overlap_tokens: int = 50

    # --- HTTP Client ---
    httpx_timeout: float = 120.0
    httpx_max_connections: int = 600
    httpx_max_keepalive_connections: int = 300

    # --- Extraction & Summary ---
    entity_extract_max_gleaning: int = int(os.environ.get("ENTITY_EXTRACT_MAX_GLEANING", "1"))
    entity_extract_max_tokens: int = int(os.environ.get("ENTITY_EXTRACT_MAX_TOKENS", "8192"))
    entity_extract_max_entities: int = int(os.environ.get("ENTITY_EXTRACT_MAX_ENTITIES", "999"))
    entity_extract_max_relationships: int = int(os.environ.get("ENTITY_EXTRACT_MAX_RELATIONSHIPS", "999"))
    entity_extract_pack_token_size: int = int(os.environ.get("ENTITY_EXTRACT_PACK_TOKEN_SIZE", "0"))
    entity_summary_to_max_tokens: int = 500

    # --- Concurrency & Storage ---
    async_semaphore_limit: int = int(os.environ.get("ASYNC_SEMAPHORE_LIMIT", "200"))
    embedding_async_semaphore_limit: int = int(os.environ.get("EMBEDDING_ASYNC_SEMAPHORE_LIMIT", "200"))
    embedding_batch_num: int = int(os.environ.get("EMBEDDING_BATCH_NUM", "32"))
    entity_extract_concurrency: int = int(os.environ.get("ENTITY_EXTRACT_CONCURRENCY", "50"))
    image_extract_concurrency: int = int(os.environ.get("IMAGE_EXTRACT_CONCURRENCY", "25"))

    # --- Graph ---
    node2vec_params: Dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # --- Beam Search ---
    beam_search_max_hops: int = 3
    beam_search_beam_width: int = 15
    beam_search_alpha: float = 0.5
    beam_search_beta: float = 1.5
    beam_search_decay: float = 0.9

    # --- KG Query ---
    kg_max_token_size: int = 4000
    kg_max_token_size_for_beam_search: int = 4000

    # Cross-modal fusion weights
    ent_to_chunk_wt: float = 0.5
    rel_to_chunk_wt: float = 0.5
    chunk_to_ent_wt: float = 0.5
    chunk_to_rel_wt: float = 0.5

    # Top-k for different retrieval types
    kg_etop_k: int = int(os.environ.get("KG_ETOP_K", "60"))
    kg_rtop_k: int = int(os.environ.get("KG_RTOP_K", "60"))
    kg_top_k: int = int(os.environ.get("KG_TOP_K", "60"))

    # --- VLM (Vision Language Model) ---
    vlm_model_name: str = os.environ.get("VLM_MODEL_NAME", "")
    vlm_base_url: str = os.environ.get("VLM_BASE_URL", "")
    vlm_api_key: str = os.environ.get("VLM_API_KEY", "")
    vlm_max_tokens: int = int(os.environ.get("VLM_MAX_TOKENS", "4096"))
    vlm_temperature: float = 0.3
    vlm_max_image_side: int = int(os.environ.get("VLM_MAX_IMAGE_SIDE", "0"))

    # --- CLIP / Multimodal Image Embedding ---
    image_embedding_model_name: str = os.environ.get("IMAGE_EMBEDDING_MODEL", "")
    image_embedding_dim: int = 768
    image_embedding_base_url: str = os.environ.get("IMAGE_EMBEDDING_BASE_URL", "")
    image_embedding_api_key: str = os.environ.get("IMAGE_EMBEDDING_API_KEY", "")

    # --- Image Extraction ---
    image_entity_extract_max_gleaning: int = int(os.environ.get("IMAGE_ENTITY_EXTRACT_MAX_GLEANING", "1"))
    image_context_max_chars: int = int(os.environ.get("IMAGE_CONTEXT_MAX_CHARS", "3500"))

    # --- Cross-Modal Entity Linking ---
    cross_modal_candidate_top_k: int = 20
    identity_confidence_threshold: float = 0.85
    depicts_confidence_threshold: float = 0.6

    # --- Beam Search Edge Type Weights ---
    edge_type_weights: Dict = field(
        default_factory=lambda: {
            "IDENTITY": 1.0,
            "DEPICTS": 0.7,
            "CONTAINS": 0.5,
            "VISUAL_RELATION": 0.6,
            "ADJACENT_TO_TEXT": 0.4,
        }
    )

    # --- Cross-Modal Propagation Weights (4 controlled weights) ---
    lambda_text_source: float = 0.5
    lambda_image_source: float = 0.5
    lambda_cross_identity: float = 1.0
    lambda_cross_depicts: float = 0.5

    # --- Image Retrieval ---
    img_top_k: int = int(os.environ.get("IMG_TOP_K", "60"))
    image_text_top_k: int = int(os.environ.get("IMAGE_TEXT_TOP_K", "60"))

    # --- Shared multimodal retrieval ---
    mm_sgn_context_max_tokens: int = int(os.environ.get("MM_SGN_CONTEXT_MAX_TOKENS", "5000"))
    mm_candidate_state_top_k: int = int(os.environ.get("MM_CANDIDATE_STATE_TOP_K", "5"))
    image_retrieval_visual_weight: float = float(os.environ.get("IMAGE_RETRIEVAL_VISUAL_WEIGHT", "0.45"))
    image_retrieval_candidate_k: int = int(os.environ.get("IMAGE_RETRIEVAL_CANDIDATE_K", "20"))

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)

        if self.embedding_base_url is None:
            self.embedding_base_url = self.llm_base_url
        if self.embedding_api_key is None:
            self.embedding_api_key = self.llm_api_key
        # Optional simple planner fallbacks
        if self.simple_planner_model_name and not self.simple_planner_base_url:
            if self.simple_planner_model_name.startswith(("gpt-", "o1", "o3", "o4")):
                self.simple_planner_base_url = "https://api.openai.com/v1"
            else:
                self.simple_planner_base_url = self.llm_base_url
        if self.simple_planner_model_name and not self.simple_planner_api_key:
            self.simple_planner_api_key = os.environ.get("OPENAI_API_KEY", self.llm_api_key)
        # VLM fallbacks
        if not self.vlm_model_name:
            self.vlm_model_name = self.llm_model_name
        if not self.vlm_base_url:
            self.vlm_base_url = self.llm_base_url
        if not self.vlm_api_key:
            self.vlm_api_key = self.llm_api_key
        # Image embedding fallbacks
        if not self.image_embedding_model_name:
            self.image_embedding_model_name = "clip-vit-large-patch14"
        if not self.image_embedding_base_url:
            self.image_embedding_base_url = self.embedding_base_url
        if not self.image_embedding_api_key:
            self.image_embedding_api_key = self.embedding_api_key

    @property
    def effective_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.llm_base_url

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key


settings = Settings()
