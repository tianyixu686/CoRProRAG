import asyncio
import json
import os
import re
from collections import Counter, defaultdict
from typing import Union, List, Dict, Any, Tuple, cast
from tqdm.asyncio import tqdm

from src.storage import JsonKVStorage

from src.base import (
    BaseGraphStorage,
    TextChunkSchema,
    BaseVectorStorage,
    BaseKVStorage,
    StorageNameSpace,
)
from src.config import Settings

from src.prompt import PROMPTS, GRAPH_FIELD_SEP

from src.utils import (
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    clean_str,
    is_float_regex,
    split_string_by_multi_markers,
    pack_user_ass_to_openai_messages,
    logger,
    compute_mdhash_id,
    parse_json_response,
)


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start: start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


def _strip_wrapping_quotes(value: str) -> str:
    if not isinstance(value, str):
        return value
    value = value.strip()
    while len(value) >= 2 and (
        (value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")
    ):
        value = value[1:-1].strip()
    return value


def _normalize_entity_for_match(value: str) -> str:
    value = _strip_wrapping_quotes(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or _strip_wrapping_quotes(record_attributes[0]) != "entity":
        return None
    entity_name = _strip_wrapping_quotes(clean_str(record_attributes[1].upper()))
    if not entity_name.strip():
        return None
    entity_type = _strip_wrapping_quotes(clean_str(record_attributes[2].upper()))
    entity_description = _strip_wrapping_quotes(clean_str(record_attributes[3]))
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or _strip_wrapping_quotes(record_attributes[0]) != "relationship":
        return None
    source = _strip_wrapping_quotes(clean_str(record_attributes[1].upper()))
    target = _strip_wrapping_quotes(clean_str(record_attributes[2].upper()))
    edge_description = _strip_wrapping_quotes(clean_str(record_attributes[3]))

    edge_keywords = _strip_wrapping_quotes(clean_str(record_attributes[4]))
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(prompt=use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    for field in ("modality", "image_path", "visual_description", "bbox"):
        values = [dp.get(field) for dp in nodes_data if dp.get(field) is not None]
        if values:
            node_data[field] = values[0]
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    entity_extract_max_tokens = global_config.get("entity_extract_max_tokens")

    ordered_chunks = list(chunks.items())
    pack_token_size = int(global_config.get("entity_extract_pack_token_size") or 0)
    if pack_token_size > 0:
        packed_chunks: list[tuple[str, TextChunkSchema]] = []
        current_ids: list[str] = []
        current_contents: list[str] = []
        current_tokens = 0

        def _flush_pack():
            nonlocal current_ids, current_contents, current_tokens
            if not current_ids:
                return
            packed_chunks.append(
                (
                    GRAPH_FIELD_SEP.join(current_ids),
                    {
                        "content": "\n\n".join(current_contents),
                        "tokens": current_tokens,
                        "chunk_order_index": len(packed_chunks),
                    },
                )
            )
            current_ids = []
            current_contents = []
            current_tokens = 0

        for chunk_id, chunk_dp in ordered_chunks:
            content = chunk_dp["content"].strip()
            if not content:
                continue
            token_count = int(chunk_dp.get("tokens") or len(content))
            if current_ids and current_tokens + token_count > pack_token_size:
                _flush_pack()
            current_ids.append(chunk_id)
            current_contents.append(f"[chunk_id: {chunk_id}]\n{content}")
            current_tokens += token_count
        _flush_pack()
        if packed_chunks:
            logger.info(
                "[Entity Extraction] Packed %d chunks into %d extraction calls "
                "(target pack token size=%d).",
                len(ordered_chunks),
                len(packed_chunks),
                pack_token_size,
            )
            ordered_chunks = packed_chunks

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        max_entities=global_config.get("entity_extract_max_entities", 999),
        max_relationships=global_config.get("entity_extract_max_relationships", 999),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0
    failed_chunks = 0

    semaphore = asyncio.Semaphore(global_config.get("entity_extract_concurrency", 50))

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations, failed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        try:
            hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
            final_result = await use_llm_func(
                prompt=hint_prompt, max_tokens=entity_extract_max_tokens
            )
            history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
            for now_glean_index in range(entity_extract_max_gleaning):
                glean_result = await use_llm_func(
                    prompt=continue_prompt,
                    history_messages=history,
                    max_tokens=entity_extract_max_tokens,
                )
                history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
                final_result += glean_result
                if now_glean_index == entity_extract_max_gleaning - 1:
                    break

                if_loop_result: str = await use_llm_func(
                    prompt=if_loop_prompt,
                    history_messages=history,
                    max_tokens=16,
                )
                if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                if if_loop_result != "yes":
                    break

            records = split_string_by_multi_markers(
                final_result,
                [context_base["record_delimiter"], context_base["completion_delimiter"]],
            )
            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(
                    record, [context_base["tuple_delimiter"]]
                )
                if_entities = await _handle_single_entity_extraction(
                    record_attributes, chunk_key
                )
                if if_entities is not None:
                    maybe_nodes[if_entities["entity_name"]].append(if_entities)
                    continue
                if_relation = await _handle_single_relationship_extraction(
                    record_attributes, chunk_key
                )
                if if_relation is not None:
                    maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                        if_relation
                    )
        except Exception as e:
            failed_chunks += 1
            logger.warning(
                "Entity extraction failed for chunk %s; skipping this chunk. Error: %s",
                chunk_key,
                e,
            )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated), {failed_chunks} failed\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    async def _process_single_content_with_sem(
        chunk_key_dp: tuple[str, TextChunkSchema],
    ):
        async with semaphore:
            return await _process_single_content(chunk_key_dp)

    results = await tqdm.gather(
        *[_process_single_content_with_sem(c) for c in ordered_chunks],
        desc="Processing chunks",
    )

    print()
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)

    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                           + dp["src_id"]
                           + dp["tgt_id"]
                           + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


# ──────────────────────────────────────────────
# Image2Graph: Visual Entity & Relation Extraction
# ──────────────────────────────────────────────

async def extract_image_graph(
    image_path: str,
    text_context: str,
    chunk_key: str,
    vlm_model_func: callable,
    global_config: dict,
) -> Tuple[Dict[str, list], Dict[tuple, list], str]:
    """Process a single image with VLM to extract visual entities and relations.

    Uses structured entity naming:
    - "IMAGE:{image_id}" — global image node
    - "VISUAL_ENTITY:{image_id}:<local_id>" — per-image visual entities

    Reuses the existing tuple format and parsing logic from text extraction.

    Returns:
        nodes: {entity_name: [entity_dict, ...]}
        edges: {(src, tgt): [edge_dict, ...]}
        global_description: image caption for storage
    """
    image_id = os.path.basename(image_path)
    # Use a safe image_id
    image_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", image_id)

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"] + ["IMAGE", "PRODUCT", "CHART", "TABLE", "LOGO", "TEXT_BLOCK"]),
        image_id=image_id,
    )

    prompt = PROMPTS["image_entity_relation_extraction"].format(
        **context_base, text_context=text_context
    )

    # VLM call
    result = await vlm_model_func(image_path=image_path, prompt=prompt)

    # Gleaning loop
    glean_limit = global_config.get("image_entity_extract_max_gleaning", 1)
    continue_prompt = PROMPTS["image_continue_extraction"]
    if_loop_prompt = PROMPTS["image_if_loop_extraction"]
    history = pack_user_ass_to_openai_messages(prompt, result)

    for i in range(glean_limit):
        glean = await vlm_model_func(
            image_path=image_path, prompt=continue_prompt, history_messages=history
        )
        history += pack_user_ass_to_openai_messages(continue_prompt, glean)
        result += glean

        if i == glean_limit - 1:
            break
        check = await vlm_model_func(
            image_path=image_path, prompt=if_loop_prompt, history_messages=history
        )
        if check.strip().lower() != "yes":
            break

    # Parse records — reuse existing parsing logic unchanged
    records = split_string_by_multi_markers(
        result, [context_base["record_delimiter"], context_base["completion_delimiter"]]
    )

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    global_description = ""

    for record in records:
        record_match = re.search(r"\((.*)\)", record)
        if record_match is None:
            continue
        record_str = record_match.group(1)
        attrs = split_string_by_multi_markers(record_str, [context_base["tuple_delimiter"]])

        # Reuse existing handler — works on tuple format regardless of source
        entity = await _handle_single_entity_extraction(attrs, chunk_key)
        if entity is not None:
            entity["entity_name"] = _strip_wrapping_quotes(entity["entity_name"])
            entity["entity_type"] = _strip_wrapping_quotes(entity["entity_type"])
            entity["description"] = _strip_wrapping_quotes(entity["description"])
            if entity["entity_name"].startswith("IMAGE:"):
                entity["modality"] = "image"
                entity["image_path"] = image_path
            elif entity["entity_name"].startswith("VISUAL_ENTITY:"):
                entity["modality"] = "image"
                entity["visual_description"] = entity["description"]
            maybe_nodes[entity["entity_name"]].append(entity)
            if entity["entity_name"].startswith("IMAGE:"):
                global_description = entity["description"]
            continue

        relation = await _handle_single_relationship_extraction(attrs, chunk_key)
        if relation is not None:
            relation["src_id"] = _strip_wrapping_quotes(relation["src_id"])
            relation["tgt_id"] = _strip_wrapping_quotes(relation["tgt_id"])
            relation["description"] = _strip_wrapping_quotes(relation["description"])
            relation["keywords"] = _strip_wrapping_quotes(relation["keywords"])
            key = (relation["src_id"], relation["tgt_id"])
            maybe_edges[key].append(relation)

    return dict(maybe_nodes), dict(maybe_edges), global_description


# ──────────────────────────────────────────────
# Cross-Modal Entity Linking: 3-Stage Pipeline
# ──────────────────────────────────────────────

async def generate_cross_modal_candidates(
    visual_entities: List[Dict[str, Any]],
    text_entities: List[Dict[str, Any]],
    nearby_chunk_ids: List[str],
    knowledge_graph: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    cfg: Settings,
) -> Dict[str, List[Dict[str, str]]]:
    """Stage A: Generate candidate text entities for each visual entity.

    Sources:
    1. Entities from nearby text chunks (via source_id)
    2. Entities semantically similar to visual description (via entity_vdb)
    3. KG 1-hop neighbors of above entities

    Returns: {visual_entity_name: [{"name": ..., "description": ...}, ...]}
    """
    candidates = {}

    # Build text entity name → description index
    text_entity_map = {e["entity_name"]: e for e in text_entities}
    normalized_text_entities = [
        (_normalize_entity_for_match(name), name)
        for name in text_entity_map
        if not name.startswith("VISUAL_ENTITY:")
    ]

    # Collect entities from nearby chunks
    nearby_text_entities = set()
    for chunk_id in nearby_chunk_ids:
        ents, _ = await knowledge_graph.find_by_chunk(chunk_id)
        for ent_name, _ in ents:
            if ent_name in text_entity_map:
                nearby_text_entities.add(ent_name)

    for vis_ent in visual_entities:
        vis_name = vis_ent["entity_name"]
        vis_desc = vis_ent.get("description", vis_ent.get("visual_description", ""))
        normalized_vis_desc = _normalize_entity_for_match(vis_desc)
        candidate_list_ordered = []
        candidate_set = set()

        def _add_candidate(name: str):
            if name in text_entity_map and name not in candidate_set:
                candidate_set.add(name)
                candidate_list_ordered.append(name)

        # 1. Nearby chunk entities (highest priority)
        for name in sorted(nearby_text_entities):
            _add_candidate(name)

        # 1b. Direct lexical mentions in VLM descriptions, e.g. person/brand names.
        for normalized_name, ent_name in normalized_text_entities:
            if len(normalized_name) >= 3 and normalized_name in normalized_vis_desc:
                _add_candidate(ent_name)

        # 2. Semantic similarity via entity_vdb
        if entity_vdb and vis_desc:
            try:
                similar = await entity_vdb.query(vis_desc, top_k=10)
                for item in similar:
                    ent_name = item.get("entity_name", "")
                    if ent_name in text_entity_map and not ent_name.startswith("VISUAL_ENTITY:"):
                        _add_candidate(ent_name)
            except Exception:
                pass

        # 3. KG 1-hop neighbor expansion
        for ent_name in list(candidate_list_ordered):
            try:
                neighbors = await knowledge_graph.get_neighbors(ent_name)
                for n in (neighbors or []):
                    if n in text_entity_map:
                        _add_candidate(n)
            except Exception:
                continue

        # Truncate to top_k
        candidate_list = [
            {
                "name": name,
                "description": text_entity_map.get(name, {}).get("description", ""),
            }
            for name in candidate_list_ordered[:cfg.cross_modal_candidate_top_k]
        ]
        candidates[vis_name] = candidate_list

    return candidates


async def align_visual_entities_to_text_entities(
    visual_entities: List[Dict[str, Any]],
    candidates: Dict[str, List[Dict[str, str]]],
    nearby_text: str,
    image_caption: str,
    llm_model_func: callable,
    cfg: Settings,
) -> List[Dict[str, Any]]:
    """Stage B: Constrained LLM judge — select from candidates only.

    The LLM sees: visual entity description + candidate text entities + nearby text.
    It must pick from the candidate list or return NO_MATCH.
    """
    alignments = []
    semaphore = asyncio.Semaphore(25)

    async def _align_single(vis_ent):
        vis_name = vis_ent["entity_name"]
        vis_desc = vis_ent.get("description", vis_ent.get("visual_description", ""))
        cands = candidates.get(vis_name, [])

        if not cands:
            return {
                "visual_entity": vis_name,
                "decision": "NO_MATCH",
                "edge_type": "",
                "confidence": 0.0,
                "evidence": "No candidates available for alignment.",
            }

        allowed_names = {c["name"] for c in cands}
        candidates_json = json.dumps(cands, indent=2)

        async with semaphore:
            prompt = PROMPTS["cross_modal_candidate_alignment"].format(
                visual_entity_name=vis_name,
                visual_description=vis_desc,
                nearby_text=nearby_text,
                image_caption=image_caption,
                candidates_json=candidates_json,
            )
            result = await llm_model_func(
                prompt=prompt,
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            try:
                parsed = await parse_json_response(result, llm_model_func)
            except Exception as exc:
                logger.warning(
                    "[CMEL] Failed to parse alignment JSON for %s; treating as NO_MATCH. Error: %s",
                    vis_name,
                    exc,
                )
                return {
                    "visual_entity": vis_name,
                    "decision": "NO_MATCH",
                    "edge_type": "",
                    "confidence": 0.0,
                    "evidence": "Alignment JSON could not be parsed.",
                }
            if parsed.get("decision") not in allowed_names and parsed.get("decision") != "NO_MATCH":
                parsed["decision"] = "NO_MATCH"
                parsed["confidence"] = 0.0
                parsed["evidence"] = "Model selected an entity outside the constrained candidate set."
            if parsed.get("decision") == "NO_MATCH":
                parsed["edge_type"] = ""
            elif parsed.get("edge_type") not in {"IDENTITY", "DEPICTS"}:
                parsed["edge_type"] = "DEPICTS"
            try:
                parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
            except (TypeError, ValueError):
                parsed["confidence"] = 0.0
            evidence = str(parsed.get("evidence", "")).strip()
            parsed["evidence"] = " ".join(evidence.split()[:40])
            return parsed

    tasks = [_align_single(ve) for ve in visual_entities]
    results = await tqdm.gather(*tasks, desc="Cross-modal alignment")

    for r in results:
        if r:
            alignments.append(r)

    return alignments


async def create_cross_modal_edges(
    alignments: List[Dict[str, Any]],
    knowledge_graph: BaseGraphStorage,
    global_config: dict,
    cfg: Settings,
    image_chunk_id: str = "",
) -> Tuple[List[Dict], List[Dict]]:
    """Stage C: Create IDENTITY/DEPICTS edges based on confidence thresholds.

    Confidence ≥ 0.85 → IDENTITY
    0.6 ≤ confidence < 0.85 → DEPICTS
    confidence < 0.6 → no edge
    """
    identity_edges = []
    depicts_edges = []

    for a in alignments:
        decision = a.get("decision", "NO_MATCH")
        confidence = a.get("confidence", 0.0)
        edge_type = a.get("edge_type", "")
        vis_entity = a.get("visual_entity", "")
        evidence = a.get("evidence", "")

        if decision == "NO_MATCH" or confidence < cfg.depicts_confidence_threshold:
            continue

        # Override edge_type based on confidence thresholds
        if confidence >= cfg.identity_confidence_threshold:
            effective_type = "IDENTITY"
        else:
            effective_type = edge_type or "DEPICTS"

        edge_src = vis_entity
        edge_tgt = decision
        edge_desc = evidence or f"Cross-modal alignment: {vis_entity} matches {decision}"
        edge_kw = "identity equivalence" if effective_type == "IDENTITY" else "likely depicts"
        edge_weight = 10.0 if effective_type == "IDENTITY" else 7.0
        edge_source_id = image_chunk_id or vis_entity

        # Add edge to KG
        try:
            await _merge_edges_then_upsert(
                edge_src, edge_tgt,
                [{"src_id": edge_src, "tgt_id": edge_tgt,
                  "weight": edge_weight, "description": edge_desc,
                  "keywords": edge_kw, "source_id": edge_source_id}],
                knowledge_graph, global_config
            )
        except Exception:
            # Fallback: use direct upsert if merge fails
            await knowledge_graph.upsert_edge(edge_src, edge_tgt, edge_data={
                "weight": edge_weight,
                "description": edge_desc,
                "keywords": edge_kw,
                "source_id": edge_source_id,
            })

        edge_data = {
            "visual_entity": vis_entity,
            "text_entity": decision,
            "edge_type": effective_type,
            "confidence": confidence,
            "evidence": evidence,
            "source_id": edge_source_id,
        }

        if effective_type == "IDENTITY":
            identity_edges.append(edge_data)
        else:
            depicts_edges.append(edge_data)

    return identity_edges, depicts_edges


async def build_multimodal_source_index(
    image_chunk_id: str,
    visual_entities: Dict[str, list],
    visual_relations: Dict[tuple, list],
    knowledge_graph: BaseGraphStorage,
):
    """Extend source indexing to the visual modality.

    Ea_img(image): image chunk → visual entities
    Ra_img(image): image chunk → visual relations
    These indexes are built automatically via source_id in _merge_* functions.
    """
    pass  # source_id propagation in _merge_nodes/edges handles this


async def merge_multimodal_graph(
    image_entities: Dict[str, list],
    image_edges: Dict[tuple, list],
    text_entities: List[Dict],
    knowledge_graph: BaseGraphStorage,
    global_config: dict,
):
    """Merge image-derived entities and edges into the shared KG.

    Reuses _merge_nodes_then_upsert and _merge_edges_then_upsert unchanged.
    Visual entities (VISUAL_ENTITY:*) never merge with text entities
    because they have different structured names.
    """
    # Merge visual entities (IMAGE:* and VISUAL_ENTITY:*)
    if image_entities:
        await asyncio.gather(*[
            _merge_nodes_then_upsert(k, v, knowledge_graph, global_config)
            for k, v in image_entities.items()
        ])
        logger.info(f"Merged {len(image_entities)} image-derived entities into KG.")

    # Merge visual relations (CONTAINS and VISUAL_RELATION)
    if image_edges:
        await asyncio.gather(*[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph, global_config)
            for k, v in image_edges.items()
        ])
        logger.info(f"Merged {len(image_edges)} image-derived relations into KG.")
async def process_files_pipeline(
    texts: List[str],
    *,
    cfg: Settings,
    full_docs_db: BaseKVStorage[TextChunkSchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    chunks_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    knowledge_graph: BaseGraphStorage,
    llm_model_func: callable,
    # ── Image modality (all optional for backward compat) ──
    images_by_doc: List[List[Dict[str, Any]]] = None,
    vlm_model_func: callable = None,
    image_embedding_func: callable = None,
    images_vdb: BaseVectorStorage = None,
    image_text_vdb: BaseVectorStorage = None,
    image_chunks_db: BaseKVStorage = None,
    visual_entities_db: BaseKVStorage = None,
    cross_modal_links_db: BaseKVStorage = None,
):
    """Process texts and images: chunk, vectorize, extract multimodal KG."""
    storage_list_for_cleanup = [
        s for s in [
            full_docs_db, text_chunks_db, chunks_vdb,
            entities_vdb, relationships_vdb, knowledge_graph,
            images_vdb, image_text_vdb, image_chunks_db,
            visual_entities_db, cross_modal_links_db,
        ] if s is not None
    ]

    try:
        # Strip and filter empty texts
        text_contents = [c.strip() for c in texts if c.strip()]

        if not text_contents:
            logger.warning("No non-empty text content provided. Aborting.")
            return None

        # Generate doc IDs and filter existing docs
        new_docs = {
            compute_mdhash_id(c, prefix="doc-"): {"content": c} for c in text_contents
        }
        _add_doc_keys = await full_docs_db.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}

        if not new_docs:
            logger.warning("All documents already exist. Aborting.")
            return None

        logger.info(f"[New Docs] Inserting {len(new_docs)} new documents.")

        # Chunk new documents
        inserting_chunks: Dict[str, Dict[str, Any]] = {}
        for doc_key, doc in new_docs.items():
            chunks_data = chunking_by_token_size(
                doc["content"],
                max_token_size=cfg.chunk_token_size,
                overlap_token_size=cfg.chunk_overlap_token_size,
                tiktoken_model=cfg.tiktoken_model_name,
            )
            chunks = {
                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                    **dp,
                    "full_doc_id": doc_key,
                }
                for dp in chunks_data
            }
            inserting_chunks.update(chunks)

        # Filter existing chunks
        _add_chunk_keys = await text_chunks_db.filter_keys(
            list(inserting_chunks.keys())
        )
        inserting_chunks = {
            k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
        }

        if not inserting_chunks:
            logger.warning("All chunks already exist. Aborting.")
            return None

        logger.info(f"[New Chunks] Inserting {len(inserting_chunks)} new chunks.")

        # Upsert chunks into vector DB
        await chunks_vdb.upsert(inserting_chunks)
        logger.info("[Vector DB] Upserted chunks into the vector database.")

        # Extract entities and relationships
        logger.info(
            "[Entity Extraction] Starting entity and relationship extraction..."
        )

        global_config = {
            "llm_model_func": llm_model_func,
            "entity_extract_max_gleaning": cfg.entity_extract_max_gleaning,
            "entity_extract_max_tokens": cfg.entity_extract_max_tokens,
            "entity_extract_max_entities": cfg.entity_extract_max_entities,
            "entity_extract_max_relationships": cfg.entity_extract_max_relationships,
            "entity_extract_pack_token_size": cfg.entity_extract_pack_token_size,
            "entity_extract_concurrency": cfg.entity_extract_concurrency,
            "entity_summary_to_max_tokens": cfg.entity_summary_to_max_tokens,
            "llm_model_max_token_size": cfg.llm_model_max_token_size,
            "tiktoken_model_name": cfg.tiktoken_model_name,
        }

        updated_knowledge_graph = await extract_entities(
            chunks=inserting_chunks,
            knowledge_graph_inst=knowledge_graph,
            entity_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            global_config=global_config,
        )

        if updated_knowledge_graph is None:
            logger.warning("Knowledge graph extraction failed or returned no result.")

        # ─────────────────────────────────────────────
        # Image Processing Pipeline (MVP)
        # ─────────────────────────────────────────────
        has_images = (
            images_by_doc is not None
            and any(imgs for imgs in images_by_doc)
            and vlm_model_func is not None
        )

        if has_images:
            logger.info("[Image Extraction] Starting image processing...")
            total_images = sum(len(imgs) for imgs in images_by_doc if imgs)
            logger.info(f"Processing {total_images} images across {len(texts)} documents.")

            image_global_config = {
                "llm_model_func": llm_model_func,
                "image_entity_extract_max_gleaning": cfg.image_entity_extract_max_gleaning,
                "entity_extract_max_gleaning": cfg.entity_extract_max_gleaning,
                "entity_summary_to_max_tokens": cfg.entity_summary_to_max_tokens,
                "llm_model_max_token_size": cfg.llm_model_max_token_size,
                "tiktoken_model_name": cfg.tiktoken_model_name,
            }

            doc_keys = list(new_docs.keys())

            # Collect all text entities for candidate generation
            all_text_entities = []
            # Get entities from KG that are text-derived (not IMAGE/VISUAL_ENTITY prefixed)
            for chunk_id in inserting_chunks:
                ents, _ = await knowledge_graph.find_by_chunk(chunk_id)
                for ent_name, ent_data in ents:
                    if not ent_name.startswith("IMAGE:") and not ent_name.startswith("VISUAL_ENTITY:"):
                        all_text_entities.append({"entity_name": ent_name, **ent_data})

            # Deduplicate
            seen = set()
            deduped_text_entities = []
            for e in all_text_entities:
                if e["entity_name"] not in seen:
                    seen.add(e["entity_name"])
                    deduped_text_entities.append(e)

            all_alignments = []
            all_cross_modal_edges = []
            all_image_entities = defaultdict(list)
            all_image_edges = defaultdict(list)
            image_chunks_data = {}
            image_text_vdb_data = {}

            semaphore = asyncio.Semaphore(cfg.image_extract_concurrency)

            def _limit_image_context(text: str) -> str:
                text = (text or "").strip()
                max_chars = max(0, int(cfg.image_context_max_chars))
                if max_chars and len(text) > max_chars:
                    return text[:max_chars].rsplit(" ", 1)[0].strip()
                return text

            async def _process_single_image(img_info, doc_idx, text_context):
                img_path = img_info.get("path", "")
                if not img_path or not os.path.exists(img_path):
                    logger.warning(f"Image not found or no path: {img_path}")
                    return None

                local_text_context = (
                    img_info.get("nearby_text")
                    or img_info.get("caption")
                    or text_context
                )
                local_text_context = _limit_image_context(local_text_context)
                img_chunk_id = compute_mdhash_id(img_path, prefix="img-")
                async with semaphore:
                    try:
                        nodes, edges, global_desc = await extract_image_graph(
                            img_path, local_text_context, img_chunk_id,
                            vlm_model_func, image_global_config,
                        )
                    except Exception as e:
                        logger.warning(
                            "Image extraction failed for %s; skipping this image. Error: %s",
                            img_path,
                            e,
                        )
                        return None
                return {
                    "img_chunk_id": img_chunk_id,
                    "nodes": nodes,
                    "edges": edges,
                    "global_desc": global_desc,
                    "img_path": img_path,
                    "nearby_text": local_text_context,
                    "caption": img_info.get("caption", global_desc),
                    "page": img_info.get("page", 0),
                    "doc_idx": doc_idx,
                }

            tasks = []
            for doc_idx, image_list in enumerate(images_by_doc or []):
                if not image_list:
                    continue
                text_context = new_docs.get(doc_keys[doc_idx], {}).get("content", "")
                for img_info in image_list:
                    tasks.append(_process_single_image(img_info, doc_idx, text_context))

            if tasks:
                results = await tqdm.gather(*tasks, desc="Processing images")
                results = [r for r in results if r is not None]

                logger.info(f"Processed {len(results)} images. Merging into KG...")

                for r in results:
                    img_chunk_id = r["img_chunk_id"]
                    # Collect all image entities and edges
                    for k, v in r["nodes"].items():
                        all_image_entities[k].extend(v)
                    for k, v in r["edges"].items():
                        all_image_edges[k].extend(v)

                    # Image chunk metadata
                    image_chunks_data[img_chunk_id] = {
                        "image_path": r["img_path"],
                        "caption": r["caption"],
                        "page": r["page"],
                        "global_description": r["global_desc"],
                        "nearby_chunk_ids": [],
                        "ocr_text": "",
                        "full_doc_id": doc_keys[r["doc_idx"]],
                        "chunk_order_index": r["doc_idx"],
                    }

                    # Image text content for image_text_vdb
                    text_content = f"{r['global_desc']} {r['caption']} {r.get('nearby_text', '')}"
                    image_text_vdb_data[img_chunk_id] = {
                        "content": text_content,
                        "image_path": r["img_path"],
                    }

                # ── Merge image entities/edges into shared KG ──
                await merge_multimodal_graph(
                    dict(all_image_entities), dict(all_image_edges),
                    deduped_text_entities, knowledge_graph, image_global_config,
                )

                # ── Cross-Modal Entity Linking ──
                # Collect visual entities for alignment
                visual_entity_list = []
                for ent_name, ent_list in all_image_entities.items():
                    if ent_name.startswith("VISUAL_ENTITY:"):
                        visual_entity_list.append({
                            "entity_name": ent_name,
                            "description": ent_list[0].get("description", ""),
                            "source_id": ent_list[0].get("source_id", ""),
                        })

                if visual_entity_list and deduped_text_entities:
                    logger.info(f"[CMEL] Aligning {len(visual_entity_list)} visual entities...")
                    # Build nearby chunk IDs from image metadata
                    for r in results:
                        image_visual_entities = [
                            ve for ve in visual_entity_list
                            if ve.get("source_id") == r["img_chunk_id"]
                        ]
                        if not image_visual_entities:
                            continue
                        nearby_chunk_ids = []
                        # Find text chunks from the same document
                        doc_id = doc_keys[r["doc_idx"]]
                        for cid, cdata in inserting_chunks.items():
                            if cdata.get("full_doc_id") == doc_id:
                                nearby_chunk_ids.append(cid)

                        # Stage A: Candidate generation
                        candidates = await generate_cross_modal_candidates(
                            image_visual_entities, deduped_text_entities,
                            nearby_chunk_ids, knowledge_graph, entities_vdb, cfg,
                        )

                        # Stage B: Alignment
                        alignments = await align_visual_entities_to_text_entities(
                            image_visual_entities, candidates,
                            r.get("nearby_text", ""), r["global_desc"],
                            llm_model_func, cfg,
                        )

                        # Stage C: Edge creation
                        id_edges, dep_edges = await create_cross_modal_edges(
                            alignments, knowledge_graph, image_global_config, cfg,
                            image_chunk_id=r["img_chunk_id"],
                        )
                        all_alignments.extend(alignments)
                        all_cross_modal_edges.extend(id_edges + dep_edges)

                        logger.info(
                            f"[CMEL] Created {len(id_edges)} IDENTITY, "
                            f"{len(dep_edges)} DEPICTS edges for {r['img_path']}"
                        )

                # ── Store image chunks ──
                if image_chunks_db and image_chunks_data:
                    await image_chunks_db.upsert(image_chunks_data)

                # ── Store visual entities for direct KG seed retrieval ──
                if entities_vdb and all_image_entities:
                    image_entity_vdb_data = {}
                    visual_entities_kv_data = {}
                    for ent_name, ent_list in all_image_entities.items():
                        ent = ent_list[0]
                        desc = ent.get("description", "")
                        image_entity_vdb_data[compute_mdhash_id(ent_name, prefix="ent-")] = {
                            "content": ent_name + desc,
                            "entity_name": ent_name,
                        }
                        if ent_name.startswith("VISUAL_ENTITY:"):
                            visual_entities_kv_data[ent_name] = {
                                "entity_name": ent_name,
                                "entity_type": ent.get("entity_type", ""),
                                "image_id": ent_name.split(":")[1] if ":" in ent_name else "",
                                "visual_description": desc,
                                "bbox": ent.get("bbox"),
                                "source_ids": ent.get("source_id", ""),
                                "modality": "image",
                            }
                    await entities_vdb.upsert(image_entity_vdb_data)
                    if visual_entities_db and visual_entities_kv_data:
                        await visual_entities_db.upsert(visual_entities_kv_data)

                # ── Store visual and cross-modal relation vectors ──
                if relationships_vdb and (all_image_edges or all_cross_modal_edges):
                    image_rel_vdb_data = {}
                    for edge_items in all_image_edges.values():
                        edge = edge_items[0]
                        rel_id = compute_mdhash_id(edge["src_id"] + edge["tgt_id"], prefix="rel-")
                        image_rel_vdb_data[rel_id] = {
                            "src_id": edge["src_id"],
                            "tgt_id": edge["tgt_id"],
                            "content": edge.get("keywords", "")
                                       + edge["src_id"]
                                       + edge["tgt_id"]
                                       + edge.get("description", ""),
                        }
                    for edge in all_cross_modal_edges:
                        src = edge["visual_entity"]
                        tgt = edge["text_entity"]
                        rel_id = compute_mdhash_id(src + tgt, prefix="rel-")
                        image_rel_vdb_data[rel_id] = {
                            "src_id": src,
                            "tgt_id": tgt,
                            "content": edge["edge_type"] + src + tgt + edge.get("evidence", ""),
                        }
                    await relationships_vdb.upsert(image_rel_vdb_data)

                # ── Store image vectors ──
                if images_vdb and image_chunks_data:
                    if image_embedding_func:
                        try:
                            await images_vdb.upsert({
                                cid: {
                                    "content": data["image_path"],
                                    "image_path": data["image_path"],
                                    "full_doc_id": data["full_doc_id"],
                                }
                                for cid, data in image_chunks_data.items()
                            })
                        except Exception as e:
                            logger.warning(f"Failed to upsert images_vdb: {e}")

                if image_text_vdb and image_text_vdb_data:
                    await image_text_vdb.upsert(image_text_vdb_data)

                # ── Store cross-modal links ──
                if cross_modal_links_db and all_alignments:
                    links_data = {
                        compute_mdhash_id(f"{a['visual_entity']}_{a.get('decision', '')}", prefix="cml-"): a
                        for a in all_alignments
                    }
                    await cross_modal_links_db.upsert(links_data)

                logger.info("[Image Processing] Complete.")

        # Persist new docs and chunks to KV storage
        await full_docs_db.upsert(new_docs)
        await text_chunks_db.upsert(inserting_chunks)
        logger.info("[KV Storage] Upserted new documents and text chunks metadata.")

        return updated_knowledge_graph

    finally:
        # Finalize all storage instances
        tasks = []
        for storage_inst in storage_list_for_cleanup:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
