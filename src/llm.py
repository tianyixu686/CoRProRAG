import os
import asyncio
import base64
import io
import mimetypes
import itertools
import numpy as np
from typing import List
import httpx
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, Timeout
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.clients import httpx_client, openai_client, openai_embedding_client
from src.config import settings
from src.base import BaseKVStorage
from src.utils import compute_args_hash, wrap_embedding_func_with_attrs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@retry(
    stop=stop_after_attempt(12),
    wait=wait_exponential(multiplier=2, min=8, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    temperature=0.7,
    max_tokens=None,
    top_p=None,
    presence_penalty=None,
    extra_body=None,
    openai_client: AsyncOpenAI = openai_client,
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    bypass_cache = kwargs.pop("bypass_cache", False)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    api_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    # Add optional params
    if max_tokens is not None:
        api_params["max_tokens"] = max_tokens
    if top_p is not None:
        api_params["top_p"] = top_p
    if presence_penalty is not None:
        api_params["presence_penalty"] = presence_penalty
    if extra_body is not None:
        api_params["extra_body"] = extra_body

    api_params.update(kwargs)

    # Check cache
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = None if bypass_cache else await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_client.chat.completions.create(**api_params)

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(12),
    wait=wait_exponential(multiplier=2, min=8, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, Timeout)
    ),
)
async def openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
    client: AsyncOpenAI = openai_embedding_client,
    **kwargs,
) -> np.ndarray:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    hashing_kv: BaseKVStorage = kwargs.pop("embedding_cache", None)
    # Check cache
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, texts)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return np.array(if_cache_return["return"])

    response = await client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )

    result = np.array([dp.embedding for dp in response.data])

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": result.tolist(), "model": model}}
        )
    return result


async def rerank_texts(
    query: str,
    texts: List[str],
    client: httpx.AsyncClient = httpx_client,
    model: str = settings.rerank_model_name,
    base_url: str = settings.rerank_base_url,
    max_retries: int = settings.rerank_max_retries,
    max_chunks_per_doc: int = settings.rerank_max_chunks_per_doc,
    overlap_tokens: int = settings.rerank_overlap_tokens,
    **kwargs,
) -> List[dict]:
    """Rerank texts using a rerank model. Returns sorted results with relevance_score and index."""

    # Check cache
    rerank_cache: BaseKVStorage = kwargs.pop("rerank_cache", None)
    if rerank_cache is not None:
        cache_key_data = {
            "query": query,
            "texts": texts,
            "model": model,
            "max_chunks_per_doc": max_chunks_per_doc,
            "overlap_tokens": overlap_tokens,
        }
        args_hash = compute_args_hash("rerank", cache_key_data)
        if_cache_return = await rerank_cache.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    payload = {
        "model": model,
        "query": query,
        "documents": texts,
        "top_n": len(texts),
        "return_documents": False,
        "max_chunks_per_doc": max_chunks_per_doc,
        "overlap_tokens": overlap_tokens,
    }

    for attempt in range(max_retries):
        try:
            headers = {
                "Authorization": f"Bearer {settings.rerank_api_key}",
                "Content-Type": "application/json",
            }
            response = await client.post(base_url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            if "results" not in result:
                raise ValueError("API response missing 'results' field")

            result = sorted(result["results"], key=lambda x: x["index"])

            # Store to cache
            if rerank_cache is not None:
                await rerank_cache.upsert(
                    {args_hash: {"return": result, "model": model}}
                )

            return result

        except Exception as e:
            # Exponential backoff, max 60s
            wait_time = min(2**attempt, 60)
            # Fixed wait for 429 rate limit
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                wait_time = 45
                print(
                    f"[Retry {attempt + 1}/{max_retries}] 429 Too Many Requests, waiting {wait_time}s..."
                )
            else:
                print(
                    f"[Retry {attempt + 1}/{max_retries}] {type(e).__name__}: {str(e)}, waiting {wait_time}s..."
                )
                print(f"Query: {query}")

            if attempt == max_retries - 1:
                print("Max retries reached, returning default scores.")
                default_result = [
                    {"relevance_score": 0.5, "index": i} for i in range(len(texts))
                ]
                # Cache default result
                if rerank_cache is not None:
                    await rerank_cache.upsert(
                        {args_hash: {"return": default_result, "model": model}}
                    )
                return default_result

            await asyncio.sleep(wait_time)


# ──────────────────────────────────────────────
# Vision Language Model (VLM)
# ──────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=2, min=8, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def vlm_complete_if_cache(
    image_path: str,
    prompt: str,
    model: str = None,
    system_prompt: str = None,
    history_messages: list = [],
    base_url: str = None,
    api_key: str = None,
    temperature: float = None,
    max_tokens: int = None,
    **kwargs,
) -> str:
    """Call a vision-language model with an image + text prompt. OpenAI-compatible API.

    The image is read from disk, encoded as base64, and sent as a data URL.
    Results are cached by (model, image_path, messages_hash).
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    # Read and encode image. Optionally downscale for VLM servers with tight
    # multimodal context budgets; this does not modify the source image.
    image_bytes = None
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    max_side = int(getattr(settings, "vlm_max_image_side", 0) or 0)
    if max_side > 0:
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                scale = min(1.0, max_side / max(w, h))
                if scale < 1.0:
                    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85, optimize=True)
                image_bytes = buf.getvalue()
                mime_type = "image/jpeg"
        except Exception:
            image_bytes = None
    if image_bytes is None:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{image_data}"

    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ],
    })

    use_model = model or settings.vlm_model_name
    use_base_url = base_url or settings.vlm_base_url or settings.llm_base_url
    use_api_key = api_key or settings.vlm_api_key or settings.llm_api_key
    use_temperature = temperature if temperature is not None else settings.vlm_temperature
    use_max_tokens = max_tokens if max_tokens is not None else settings.vlm_max_tokens

    # Cache check
    if hashing_kv is not None:
        args_hash = compute_args_hash(use_model, image_path, messages, use_temperature)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    client = AsyncOpenAI(
        base_url=use_base_url,
        api_key=use_api_key,
        http_client=httpx.AsyncClient(trust_env=False),
    )
    response = await client.chat.completions.create(
        model=use_model,
        messages=messages,
        temperature=use_temperature,
        max_tokens=use_max_tokens,
    )

    result = response.choices[0].message.content

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": result, "model": use_model}}
        )

    return result


# ──────────────────────────────────────────────
# CLIP / Multimodal Image Embedding
# ──────────────────────────────────────────────

@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=512)
@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=2, min=8, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def clip_image_embedding(
    texts: list[str],
    model: str = None,
    mode: str = "image",
    base_url: str = None,
    api_key: str = None,
    **kwargs,
) -> np.ndarray:
    """CLIP dual-tower embedding.

    mode="image": texts are image file paths → CLIP image encoder → 768-dim.
    mode="text":  texts are query strings → CLIP text encoder → 768-dim.
    Both modes map to the same CLIP latent space for cross-modal retrieval.

    Uses an OpenAI-compatible embeddings API endpoint.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    embedding_cache: BaseKVStorage = kwargs.pop("embedding_cache", None)
    use_model = model or settings.image_embedding_model_name
    use_base_url = base_url or settings.image_embedding_base_url or settings.llm_base_url
    use_api_key = api_key or settings.image_embedding_api_key or settings.llm_api_key

    # Cache check
    if embedding_cache is not None:
        args_hash = compute_args_hash(use_model, texts, mode)
        if_cache_return = await embedding_cache.get_by_id(args_hash)
        if if_cache_return is not None:
            return np.array(if_cache_return["return"])

    client = AsyncOpenAI(
        base_url=use_base_url,
        api_key=use_api_key,
        http_client=httpx.AsyncClient(trust_env=False),
    )

    if mode == "image":
        # Encode images as base64 data URLs for CLIP image encoder
        inputs = []
        for path in texts:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            inputs.append(f"data:{mime};base64,{b64}")
        response = await client.embeddings.create(
            model=use_model, input=inputs, encoding_format="float"
        )
    else:
        # "text" mode: encode through CLIP text encoder
        response = await client.embeddings.create(
            model=use_model, input=texts, encoding_format="float"
        )

    result = np.array([dp.embedding for dp in response.data])

    if embedding_cache is not None:
        await embedding_cache.upsert(
            {args_hash: {"return": result.tolist(), "model": use_model}}
        )

    return result
