import asyncio
import json
from functools import partial
from typing import List, Any, Dict

import numpy as np
from openai import AsyncOpenAI

from src.utils import EmbeddingFunc, logger
from src.llm import (
    openai_complete_if_cache,
    openai_embedding,
    rerank_texts,
    vlm_complete_if_cache,
    clip_image_embedding,
)
from src.storage import (
    JsonKVStorage,
    NanoImageVectorDBStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)

from src.kg_builder import process_files_pipeline
from src.config import Settings, settings
from src.prompt import PROMPTS

from src.pipeline.query_executor import cor_simple_query


class App:
    """Application container and main entry point."""

    def __init__(self, cfg: Settings):
        self.config = cfg

        # Init concurrency and cache
        self.async_semaphore = asyncio.Semaphore(self.config.async_semaphore_limit)
        self.embedding_async_semaphore = asyncio.Semaphore(
            self.config.embedding_async_semaphore_limit
        )
        self.llm_cache_db = self._init_llm_cache_db()
        self.embedding_cache_db = self._init_embedding_cache_db()
        self.rerank_cache_db = self._init_rerank_cache_db()

        # Init services (depends on cache)
        self.llm_model_func = partial(
            self._raw_llm_model_func, hashing_kv=self.llm_cache_db
        )
        self.planner_openai_client = None
        self.planner_model_func = None
        if self.config.simple_planner_model_name:
            self.planner_openai_client = AsyncOpenAI(
                base_url=self.config.simple_planner_base_url or None,
                api_key=self.config.simple_planner_api_key or self.config.llm_api_key,
            )
            self.planner_model_func = partial(
                self._raw_simple_planner_model_func, hashing_kv=self.llm_cache_db
            )
        self.embedding_func = partial(
            self._raw_embedding_func, embedding_cache=self.embedding_cache_db
        )

        # Cached rerank function
        self.rerank_texts_func = partial(
            rerank_texts, rerank_cache=self.rerank_cache_db
        )

        # EmbeddingFunc wrapper for vector stores
        self.embedding_func_wrapper = EmbeddingFunc(
            embedding_dim=self.config.embedding_dim,
            max_token_size=self.config.embedding_max_token_size,
            func=self.embedding_func,
        )

        # Init all storages
        self.full_docs_db = self._init_full_docs_vdb()
        self.text_chunks_db = self._init_text_chunks_vdb()
        self.chunks_vdb = self._init_chunks_vdb()
        self.entities_vdb = self._init_entity_vdb()
        self.relationships_vdb = self._init_relationship_vdb()
        self.knowledge_graph = self._init_chunk_entity_relation_graph()

        # ── VLM model function ──
        self.vlm_model_func = partial(
            self._raw_vlm_model_func, hashing_kv=self.llm_cache_db
        )

        # ── Image embedding function ──
        self.image_embedding_func_wrapper = EmbeddingFunc(
            embedding_dim=self.config.image_embedding_dim,
            max_token_size=512,
            func=partial(
                self._raw_image_embedding_func,
                embedding_cache=self.embedding_cache_db,
            ),
        )

        # ── Image modality storages ──
        self.images_vdb = self._init_images_vdb()
        self.image_text_vdb = self._init_image_text_vdb()
        self.image_chunks_db = self._init_image_chunks_db()
        self.visual_entities_db = self._init_visual_entities_db()
        self.cross_modal_links_db = self._init_cross_modal_links_db()

        self._cache_storages = [
            self.llm_cache_db,
            self.embedding_cache_db,
            self.rerank_cache_db,
        ]
        self._all_storages = [
            s for s in self.__dict__.values() if hasattr(s, "index_done_callback")
        ]

    # ---------- LLM & Embedding ----------
    async def _raw_llm_model_func(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: List[Any] = [],
        model: str = None,
        **kwargs,
    ) -> str:
        temperature = kwargs.pop("temperature", self.config.llm_temperature)
        max_tokens = kwargs.pop("max_tokens", self.config.llm_max_tokens)
        top_p = kwargs.pop("top_p", self.config.llm_top_p)
        presence_penalty = kwargs.pop(
            "presence_penalty", self.config.llm_presence_penalty
        )
        extra_body = kwargs.pop("extra_body", self.config.llm_extra_body)

        async with self.async_semaphore:
            return await openai_complete_if_cache(
                model=model or self.config.llm_model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                extra_body=extra_body,
                **kwargs,
            )

    async def _raw_simple_planner_model_func(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: List[Any] = [],
        model: str = None,
        **kwargs,
    ) -> str:
        temperature = kwargs.pop("temperature", 0.0)
        max_tokens = kwargs.pop("max_tokens", min(self.config.llm_max_tokens or 1024, 2048))
        async with self.async_semaphore:
            return await openai_complete_if_cache(
                model=model or self.config.simple_planner_model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=self.config.simple_planner_api_key or self.config.llm_api_key,
                openai_client=self.planner_openai_client,
                **kwargs,
            )

    async def _raw_embedding_func(self, texts: List[str], **kwargs) -> np.ndarray:
        async with self.embedding_async_semaphore:
            return await openai_embedding(
                texts, model=self.config.embedding_model_name, **kwargs
            )

    async def _raw_vlm_model_func(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: List[Any] = [],
        model: str = None,
        **kwargs,
    ) -> str:
        """VLM call wrapped with semaphore and cache."""
        temperature = kwargs.pop("temperature", self.config.vlm_temperature)
        max_tokens = kwargs.pop("max_tokens", self.config.vlm_max_tokens)

        async with self.async_semaphore:
            return await vlm_complete_if_cache(
                image_path=image_path,
                prompt=prompt,
                model=model or self.config.vlm_model_name,
                system_prompt=system_prompt,
                history_messages=history_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

    async def _raw_image_embedding_func(self, texts: List[str], mode: str = "image", **kwargs) -> np.ndarray:
        """CLIP image embedding wrapped with semaphore."""
        async with self.embedding_async_semaphore:
            return await clip_image_embedding(
                texts, model=self.config.image_embedding_model_name, mode=mode, **kwargs
            )

    # ---------- Private init methods ----------
    def _init_embedding_cache_db(self) -> JsonKVStorage:
        return JsonKVStorage(
            namespace="embedding_cache",
            global_config={"working_dir": self.config.data_dir},
        )

    def _init_llm_cache_db(self) -> JsonKVStorage:
        return JsonKVStorage(
            namespace="llm_response_cache",
            global_config={"working_dir": self.config.data_dir},
        )

    def _init_rerank_cache_db(self) -> JsonKVStorage:
        return JsonKVStorage(
            namespace="rerank_cache",
            global_config={"working_dir": self.config.data_dir},
        )

    def _init_chunks_vdb(self) -> NanoVectorDBStorage:
        return NanoVectorDBStorage(
            namespace="chunks",
            global_config={
                "working_dir": self.config.data_dir,
                "embedding_batch_num": self.config.embedding_batch_num,
            },
            embedding_func=self.embedding_func_wrapper,
            meta_fields={"content", "full_doc_id"},
        )

    def _init_text_chunks_vdb(self) -> JsonKVStorage:
        return JsonKVStorage(
            namespace="text_chunks", global_config={"working_dir": self.config.data_dir}
        )

    def _init_full_docs_vdb(self) -> JsonKVStorage:
        return JsonKVStorage(
            namespace="full_docs", global_config={"working_dir": self.config.data_dir}
        )

    def _init_entity_vdb(self) -> NanoVectorDBStorage:
        return NanoVectorDBStorage(
            namespace="entities",
            global_config={
                "working_dir": self.config.data_dir,
                "embedding_batch_num": self.config.embedding_batch_num,
            },
            embedding_func=self.embedding_func_wrapper,
            meta_fields={"entity_name"},
        )

    def _init_relationship_vdb(self) -> NanoVectorDBStorage:
        return NanoVectorDBStorage(
            namespace="relationships",
            global_config={
                "working_dir": self.config.data_dir,
                "embedding_batch_num": self.config.embedding_batch_num,
            },
            embedding_func=self.embedding_func_wrapper,
            meta_fields={"src_id", "tgt_id"},
        )

    def _init_chunk_entity_relation_graph(self) -> NetworkXStorage:
        return NetworkXStorage(
            namespace="chunk_entity_relation",
            global_config={
                "working_dir": self.config.data_dir,
                "node2vec_params": self.config.node2vec_params,
            },
        )

    # ── Image modality storage init ──

    def _init_images_vdb(self) -> NanoImageVectorDBStorage:
        """CLIP image embeddings — visual semantic retrieval."""
        return NanoImageVectorDBStorage(
            namespace="images",
            global_config={
                "working_dir": self.config.data_dir,
                "embedding_batch_num": self.config.embedding_batch_num,
            },
            embedding_func=self.image_embedding_func_wrapper,
            meta_fields={"image_path", "full_doc_id"},
        )

    def _init_image_text_vdb(self) -> NanoVectorDBStorage:
        """Image caption/OCR/description — text-based image retrieval."""
        return NanoVectorDBStorage(
            namespace="image_text",
            global_config={
                "working_dir": self.config.data_dir,
                "embedding_batch_num": self.config.embedding_batch_num,
            },
            embedding_func=self.embedding_func_wrapper,
            meta_fields={"image_path"},
        )

    def _init_image_chunks_db(self) -> JsonKVStorage:
        return JsonKVStorage(
            namespace="image_chunks",
            global_config={"working_dir": self.config.data_dir},
        )

    def _init_visual_entities_db(self) -> JsonKVStorage:
        return JsonKVStorage(
            namespace="visual_entities",
            global_config={"working_dir": self.config.data_dir},
        )

    def _init_cross_modal_links_db(self) -> JsonKVStorage:
        return JsonKVStorage(
            namespace="cross_modal_links",
            global_config={"working_dir": self.config.data_dir},
        )

    # ---------- Public methods ----------
    async def insert_one_json_file(self, json_file_path: str):
        """Insert a JSON file containing a list of {title, text, images?} objects.

        Each item may optionally have an 'images' field:
        [{title, text, images: [{path, page?, caption?, nearby_text?}]}]
        """
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            texts = [f"{item['title']}\n{item['text']}" for item in data]
            images_by_doc = [item.get('images', []) for item in data]

            logger.info(f"Starting to process {len(texts)} texts...")
            if any(imgs for imgs in images_by_doc):
                total_images = sum(len(imgs) for imgs in images_by_doc if imgs)
                logger.info(f"Found {total_images} images across documents.")

            updated_knowledge_graph = await process_files_pipeline(
                texts,
                cfg=self.config,
                full_docs_db=self.full_docs_db,
                text_chunks_db=self.text_chunks_db,
                chunks_vdb=self.chunks_vdb,
                entities_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                knowledge_graph=self.knowledge_graph,
                llm_model_func=self.llm_model_func,
                images_by_doc=images_by_doc,
                vlm_model_func=self.vlm_model_func,
                image_embedding_func=self.image_embedding_func_wrapper.func,
                images_vdb=self.images_vdb,
                image_text_vdb=self.image_text_vdb,
                image_chunks_db=self.image_chunks_db,
                visual_entities_db=self.visual_entities_db,
                cross_modal_links_db=self.cross_modal_links_db,
            )

            if updated_knowledge_graph is not None:
                self.knowledge_graph = updated_knowledge_graph
                logger.info("Knowledge graph updated successfully.")
            else:
                logger.warning("Knowledge graph processing returned None.")

            logger.info("File processing finished.")

        except Exception as e:
            logger.error(f"Error during file processing: {e}")
            raise

    async def _build_query_with_image(self, query: str, query_image_path: str | None) -> str:
        if not query_image_path:
            return query
        visual_prompt = PROMPTS["query_image_understanding"].format(query=query)
        try:
            visual_context = await self.vlm_model_func(
                image_path=query_image_path,
                prompt=visual_prompt,
                max_tokens=min(self.config.vlm_max_tokens, 512),
            )
        except Exception as exc:
            logger.warning(
                "Query image VLM understanding failed; continuing with image-to-image retrieval only: %s",
                exc,
            )
            return query
        visual_context = (visual_context or "").strip()
        if not visual_context:
            return query
        return f"Question image evidence:\n{visual_context}\n\nQuestion:\n{query}"

    async def query(
        self,
        query: str,
        method: str,
        full_context: str = None,
        query_image_path: str | None = None,
    ) -> Dict:
        global_config = {
            "llm_model_func": self.llm_model_func,
            "planner_model_func": self.planner_model_func,
            "vlm_model_func": self.vlm_model_func,
            "config": self.config,
            "rerank_texts_func": self.rerank_texts_func,
        }
        if method != "cor_simple":
            raise ValueError(f"Unknown query method: {method}. Available methods: ['cor_simple']")
        return await cor_simple_query(
            query=query,
            knowledge_graph_inst=self.knowledge_graph,
            entities_vdb=self.entities_vdb,
            relationships_vdb=self.relationships_vdb,
            chunks_vdb=self.chunks_vdb,
            text_chunks_db=self.text_chunks_db,
            global_config=global_config,
            images_vdb=self.images_vdb,
            image_text_vdb=self.image_text_vdb,
            image_chunks_db=self.image_chunks_db,
            visual_entities_db=self.visual_entities_db,
            query_image_path=query_image_path,
        )

    async def close_cache_only(self):
        """Close cache storage connections only."""
        logger.info("Closing cache resources and performing cleanup...")
        tasks = [
            s.index_done_callback()
            for s in self._cache_storages
            if hasattr(s, "index_done_callback")
        ]
        await asyncio.gather(*tasks)
        logger.info("Cache cleanup complete.")

    async def close(self):
        """Close all storage connections and perform cleanup."""
        logger.info("Closing all resources and performing cleanup...")
        tasks = [s.index_done_callback() for s in self._all_storages]
        await asyncio.gather(*tasks)
        logger.info("Complete cleanup finished.")


async def main():
    app = App(settings)
    result = await app.query(
        query="Who is the mascot of the university related to Randy Conrads?",
        method="cor_simple",
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
