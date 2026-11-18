"""Fusion retrieval: combining knowledge graph and text retrieval."""

import asyncio
import json

from src.utils import (
    logger,
    parse_json_response,
    normalize_list,
)
from src.llm import rerank_texts
from src.prompt import PROMPTS


async def _extract_keywords_relationships(query, use_model_func):
    """Extract keywords and relationships from the query."""
    extraction_prompt = PROMPTS["keywords_relationship_extraction"].format(query=query)
    llm_result = await use_model_func(prompt=extraction_prompt)

    try:
        return await parse_json_response(llm_result, use_model_func)
    except json.JSONDecodeError:
        # Both extraction and JSON repair may have cached malformed responses.
        async def fresh_model(**kwargs):
            return await use_model_func(**kwargs, bypass_cache=True)
        llm_result = await fresh_model(prompt=extraction_prompt)
        try:
            return await parse_json_response(llm_result, fresh_model)
        except json.JSONDecodeError:
            print("Failed to extract keywords and relationships!")
            return None


def _normalize_keywords_relationships(query, keywords_relationships):
    """Return a retrieval-safe keyword/relationship payload."""
    def _as_string_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            for key in ("keywords", "relationships", "items", "values"):
                if key in value:
                    return _as_string_list(value[key])
            return [json.dumps(value, ensure_ascii=False)]
        if isinstance(value, (list, tuple, set)):
            result = []
            for item in value:
                result.extend(_as_string_list(item))
            return result
        return [str(value)]

    if not isinstance(keywords_relationships, dict):
        keywords_relationships = {}

    keywords = _as_string_list(keywords_relationships.get("keywords"))
    relationships = _as_string_list(keywords_relationships.get("relationships"))

    if not keywords:
        keywords = [query]

    return {
        "keywords": [str(keyword) for keyword in keywords if str(keyword).strip()],
        "relationships": [
            str(relationship)
            for relationship in relationships
            if str(relationship).strip()
        ],
    }


async def _parallel_retrieval(
    query,
    keywords_relationships,
    chunks_vdb,
    entities_vdb,
    relationships_vdb,
    top_k,
    etop_k,
):
    """Retrieve chunks, entities, and relations in parallel."""
    keywords_relationships = _normalize_keywords_relationships(
        query, keywords_relationships
    )
    keywords_str = ",".join(keywords_relationships["keywords"])
    relationships_list = keywords_relationships["relationships"]

    chunks_task = asyncio.create_task(chunks_vdb.query(query, top_k=top_k))
    entities_task = asyncio.create_task(entities_vdb.query(keywords_str, top_k=etop_k))
    relations_tasks = [
        asyncio.create_task(relationships_vdb.query(rel_query, top_k=15))
        for rel_query in relationships_list
    ]

    gathered = await asyncio.gather(chunks_task, entities_task, *relations_tasks)
    chunks_result, entities_result, *relations_results = gathered

    unique_items = {}
    for sublist in relations_results:
        for item in sublist:
            item_id = item["id"]
            if (
                item_id not in unique_items
                or item["distance"] > unique_items[item_id]["distance"]
            ):
                unique_items[item_id] = item

    return chunks_result, entities_result, list(unique_items.values())


async def _parallel_retrieval_with_rerank(
    query,
    keywords_relationships,
    chunks_vdb,
    entities_vdb,
    relationships_vdb,
    top_k,
    etop_k,
    rerank_texts_func=rerank_texts,
):
    """Retrieve chunks, entities, and relations in parallel with reranking."""
    keywords_relationships = _normalize_keywords_relationships(
        query, keywords_relationships
    )
    keywords_str = ",".join(keywords_relationships["keywords"])
    relationships_list = keywords_relationships["relationships"]

    chunks_task = asyncio.create_task(chunks_vdb.query(query, top_k=top_k))
    entities_task = asyncio.create_task(entities_vdb.query(keywords_str, top_k=etop_k))
    relations_tasks = [
        asyncio.create_task(relationships_vdb.query(rel_query, top_k=15))
        for rel_query in relationships_list
    ]

    gathered = await asyncio.gather(chunks_task, entities_task, *relations_tasks)
    chunks_result, entities_result, *relations_results = gathered

    unique_items = {}
    for sublist in relations_results:
        for item in sublist:
            item_id = item["id"]
            if (
                item_id not in unique_items
                or item["distance"] > unique_items[item_id]["distance"]
            ):
                unique_items[item_id] = item
    relations_result = list(unique_items.values())

    chunk_texts = [chunk.get("content", "") for chunk in chunks_result]
    entity_texts = [entity.get("entity_name", "") for entity in entities_result]
    relation_texts = [
        f"{rel.get('src_id', '')} -> {rel.get('tgt_id', '')}"
        for rel in relations_result
    ]

    async def empty_rerank():
        return []

    rerank_tasks = [
        rerank_texts_func(query=query, texts=chunk_texts)
        if chunk_texts
        else empty_rerank(),
        rerank_texts_func(query=query, texts=entity_texts)
        if entity_texts
        else empty_rerank(),
        rerank_texts_func(query=query, texts=relation_texts)
        if relation_texts
        else empty_rerank(),
    ]

    (
        chunk_rerank_scores,
        entity_rerank_scores,
        relation_rerank_scores,
    ) = await asyncio.gather(*rerank_tasks)

    for i, chunk in enumerate(chunks_result):
        if i < len(chunk_rerank_scores):
            chunk["__metrics__"] = chunk_rerank_scores[i].get("relevance_score", 0.0)

    for i, entity in enumerate(entities_result):
        if i < len(entity_rerank_scores):
            entity["__metrics__"] = entity_rerank_scores[i].get("relevance_score", 0.0)

    for i, relation in enumerate(relations_result):
        if i < len(relation_rerank_scores):
            relation["__metrics__"] = relation_rerank_scores[i].get(
                "relevance_score", 0.0
            )

    return chunks_result, entities_result, relations_result


def _initialize_retrieval_dicts(chunks_result, entities_result, relations_results):
    """Initialize chunk, entity, and relation score dicts."""
    text_dict = {
        chunk["id"]: {
            "id": chunk["id"],
            "weighted_score": chunk["__metrics__"],
        }
        for chunk in chunks_result[:15]
    }
    entities_dict = {
        entity["id"]: {
            "id": entity["id"],
            "entity_name": entity["entity_name"],
            "weighted_score": entity["__metrics__"],
        }
        for entity in entities_result[:20]
    }
    relationships_dict = {
        relation["id"]: {
            "id": relation["id"],
            "src_name": relation["src_id"],
            "tgt_name": relation["tgt_id"],
            "weighted_score": relation["__metrics__"],
        }
        for relation in relations_results[:5]
    }

    return text_dict, entities_dict, relationships_dict


async def _add_entity_relation_descriptions(entities, relationships, kg_graph):
    """Add descriptions for entities and relations in parallel."""

    async def get_entity_description(entity):
        entity_name = entity["entity_name"]
        entity_content = await kg_graph.get_node(entity_name)
        entity["description"] = entity_content.get("description", "") if entity_content else ""

    async def get_relationship_description(relationship):
        src_name = relationship["src_name"]
        tgt_name = relationship["tgt_name"]
        relationship_content = await kg_graph.get_edge(src_name, tgt_name)
        relationship["description"] = relationship_content.get("description", "") if relationship_content else ""

    await asyncio.gather(
        *[get_entity_description(entity) for entity in entities],
        *[get_relationship_description(relationship) for relationship in relationships],
    )


async def _compute_chunk_to_entity_contributions(
    text_dict, knowledge_graph, entity_vdb, chunk_to_ent_wt
):
    """Compute chunk-to-entity score contributions."""
    c_to_e_contrib = {}

    for cid, chunk_item in text_dict.items():
        score = chunk_item["weighted_score"]
        ents, _ = await knowledge_graph.find_by_chunk(cid)

        for entity_name, entity_attr in ents:
            eid = await entity_vdb.get_entity_id(name=entity_name)
            contrib = score * chunk_to_ent_wt
            if eid not in c_to_e_contrib:
                c_to_e_contrib[eid] = {
                    "weighted_score": contrib,
                    "entity_name": entity_name,
                }
            else:
                c_to_e_contrib[eid]["weighted_score"] += contrib

    return c_to_e_contrib


async def _compute_chunk_to_relation_contributions(
    text_dict, knowledge_graph, relationship_vdb, chunk_to_rel_wt
):
    """Compute chunk-to-relation score contributions."""
    c_to_r_contrib = {}

    for cid, chunk_item in text_dict.items():
        score = chunk_item["weighted_score"]
        _, rels = await knowledge_graph.find_by_chunk(cid)

        for src, tgt, rel_attr in rels:
            rid = await relationship_vdb.get_rel_id(src=src, tgt=tgt)
            if rid is None:
                rid = await relationship_vdb.get_rel_id(src=tgt, tgt=src)
            contrib = score * chunk_to_rel_wt
            if rid not in c_to_r_contrib:
                c_to_r_contrib[rid] = {
                    "weighted_score": contrib,
                    "src_name": src,
                    "tgt_name": tgt,
                }
            else:
                c_to_r_contrib[rid]["weighted_score"] += contrib

    return c_to_r_contrib


async def _compute_entity_to_chunk_contributions(
    entities_dict, knowledge_graph, ent_to_chunk_wt
):
    """Compute entity-to-chunk score contributions."""
    e_to_c_contrib = {}

    for eid, entity_item in entities_dict.items():
        score = entity_item["weighted_score"]
        entity_name = entity_item["entity_name"]
        node = await knowledge_graph.get_node(entity_name)
        if node is None:
            continue
        chunk_id_list = node.get("source_id", "")
        chunk_id_list = chunk_id_list.split("<SEP>") if chunk_id_list else []

        for chunk_id in chunk_id_list:
            if not chunk_id.startswith("chunk-"):
                continue
            contrib = score * ent_to_chunk_wt
            if chunk_id not in e_to_c_contrib:
                e_to_c_contrib[chunk_id] = contrib
            else:
                e_to_c_contrib[chunk_id] += contrib

    return e_to_c_contrib


async def _compute_relation_to_chunk_contributions(
    relationships_dict, knowledge_graph, rel_to_chunk_wt
):
    """Compute relation-to-chunk score contributions."""
    r_to_c_contrib = {}

    for rid, relation_item in relationships_dict.items():
        score = relation_item["weighted_score"]
        src_name = relation_item["src_name"]
        tgt_name = relation_item["tgt_name"]
        edge = await knowledge_graph.get_edge(src_name, tgt_name)
        if edge is None:
            continue
        chunk_id_list = edge.get("source_id", "")
        chunk_id_list = chunk_id_list.split("<SEP>") if chunk_id_list else []

        for chunk_id in chunk_id_list:
            if not chunk_id.startswith("chunk-"):
                continue
            contrib = score * rel_to_chunk_wt
            if chunk_id not in r_to_c_contrib:
                r_to_c_contrib[chunk_id] = contrib
            else:
                r_to_c_contrib[chunk_id] += contrib

    return r_to_c_contrib


def _apply_contributions_to_dicts(
    text_dict,
    entities_dict,
    relationships_dict,
    c_to_e_contrib,
    c_to_r_contrib,
    e_to_c_contrib,
    r_to_c_contrib,
    # ── Cross-modal (optional) ──
    img_to_ent_contrib=None,
    img_to_chunk_contrib=None,
    ent_to_img_contrib=None,
    chunk_to_img_contrib=None,
    rel_to_img_contrib=None,
    img_to_rel_contrib=None,
    images_dict=None,
):
    """Apply all cross-contributions to the score dicts, including image modality."""
    for chunk_id, contrib in e_to_c_contrib.items():
        if chunk_id not in text_dict:
            text_dict[chunk_id] = {"id": chunk_id, "weighted_score": contrib}
        else:
            text_dict[chunk_id]["weighted_score"] += contrib

    for chunk_id, contrib in r_to_c_contrib.items():
        if chunk_id not in text_dict:
            text_dict[chunk_id] = {"id": chunk_id, "weighted_score": contrib}
        else:
            text_dict[chunk_id]["weighted_score"] += contrib

    for eid, contrib in c_to_e_contrib.items():
        if eid and eid not in entities_dict:
            entities_dict[eid] = {
                "id": eid,
                "entity_name": contrib["entity_name"],
                "weighted_score": contrib["weighted_score"],
            }
        elif eid:
            entities_dict[eid]["weighted_score"] += contrib["weighted_score"]

    for rid, contrib in c_to_r_contrib.items():
        if rid not in relationships_dict:
            relationships_dict[rid] = {
                "id": rid,
                "src_name": contrib["src_name"],
                "tgt_name": contrib["tgt_name"],
                "weighted_score": contrib["weighted_score"],
            }
        else:
            relationships_dict[rid]["weighted_score"] += contrib["weighted_score"]

    # ── Cross-modal image contributions ──
    if img_to_ent_contrib and images_dict:
        for eid, contrib in img_to_ent_contrib.items():
            score = contrib.get("weighted_score", 0.0) if isinstance(contrib, dict) else contrib
            entity_name = contrib.get("entity_name", "") if isinstance(contrib, dict) else ""
            if eid and eid not in entities_dict:
                entities_dict[eid] = {
                    "id": eid,
                    "entity_name": entity_name,
                    "weighted_score": score,
                }
            elif eid:
                entities_dict[eid]["weighted_score"] += score

    if img_to_chunk_contrib and images_dict:
        for chunk_id, contrib in img_to_chunk_contrib.items():
            if chunk_id not in text_dict:
                text_dict[chunk_id] = {"id": chunk_id, "weighted_score": contrib}
            else:
                text_dict[chunk_id]["weighted_score"] += contrib

    if ent_to_img_contrib and images_dict:
        for img_id, score in ent_to_img_contrib.items():
            if img_id not in images_dict:
                images_dict[img_id] = {"id": img_id, "weighted_score": score}
            else:
                images_dict[img_id]["weighted_score"] += score

    if chunk_to_img_contrib and images_dict:
        for img_id, score in chunk_to_img_contrib.items():
            if img_id not in images_dict:
                images_dict[img_id] = {"id": img_id, "weighted_score": score}
            else:
                images_dict[img_id]["weighted_score"] += score

    if rel_to_img_contrib and images_dict:
        for img_id, score in rel_to_img_contrib.items():
            if img_id not in images_dict:
                images_dict[img_id] = {"id": img_id, "weighted_score": score}
            else:
                images_dict[img_id]["weighted_score"] += score

    if img_to_rel_contrib and images_dict:
        for rid, contrib in img_to_rel_contrib.items():
            score = contrib.get("weighted_score", 0.0) if isinstance(contrib, dict) else contrib
            if rid not in relationships_dict:
                relationships_dict[rid] = {
                    "id": rid,
                    "src_name": contrib.get("src_name", "") if isinstance(contrib, dict) else "",
                    "tgt_name": contrib.get("tgt_name", "") if isinstance(contrib, dict) else "",
                    "weighted_score": score,
                }
            else:
                relationships_dict[rid]["weighted_score"] += score


def _sort_and_truncate_dicts(text_dict, entities_dict, relationships_dict, top_k=60):
    """Sort and truncate dicts, keeping top-k entries."""
    text_dict = dict(
        sorted(text_dict.items(), key=lambda x: x[1]["weighted_score"], reverse=True)[
            :top_k
        ]
    )
    entities_dict = dict(
        sorted(
            entities_dict.items(), key=lambda x: x[1]["weighted_score"], reverse=True
        )[:top_k]
    )
    relationships_dict = dict(
        sorted(
            relationships_dict.items(),
            key=lambda x: x[1]["weighted_score"],
            reverse=True,
        )[:top_k]
    )
    return text_dict, entities_dict, relationships_dict


async def _finalize_results(
    text_dict, entities_dict, relationships_dict, text_chunks_db, knowledge_graph
):
    """Finalize results: fetch chunk content and descriptions."""
    ranked_chunks = list(text_dict.values())
    chunks_ids = [chunk["id"] for chunk in ranked_chunks[:60]]
    ranked_chunks_content = await text_chunks_db.get_by_ids(chunks_ids)
    for chunk, chunk_content in zip(ranked_chunks, ranked_chunks_content):
        chunk["content"] = chunk_content["content"] if chunk_content else ""
    ranked_chunks = [chunk for chunk in ranked_chunks if chunk.get("content")]

    ranked_entities = list(entities_dict.values())
    ranked_relationships = list(relationships_dict.values())

    await _add_entity_relation_descriptions(
        ranked_entities, ranked_relationships, knowledge_graph
    )

    return ranked_chunks, ranked_entities, ranked_relationships


async def process_retrieval_results(
    chunks_result, entities_result, relations_results, knowledge_graph
):
    """Process retrieval results: fetch details and normalize scores."""
    chunks = []
    entities = []
    relationships = []

    for chunk in chunks_result:
        chunks.append(
            {
                "id": chunk["id"],
                "weighted_score": chunk["__metrics__"],
                "content": chunk["content"],
            }
        )

    entity_name_list = [entity["entity_name"] for entity in entities_result]
    entities_with_content = await knowledge_graph.get_nodes(entity_name_list)
    for entity, entity_content in zip(entities_result, entities_with_content):
        if entity_content is None:
            continue
        entities.append(
            {
                "id": entity["id"],
                "entity_name": entity["entity_name"],
                "weighted_score": entity["__metrics__"],
                "description": entity_content["description"],
            }
        )

    relationship_key_list = [
        (relation["src_id"], relation["tgt_id"]) for relation in relations_results
    ]
    relationships_with_content = await knowledge_graph.get_edges(relationship_key_list)
    for relation, relation_content in zip(
        relations_results, relationships_with_content
    ):
        if relation_content is None:
            continue
        relationships.append(
            {
                "id": relation["id"],
                "src_name": relation["src_id"],
                "tgt_name": relation["tgt_id"],
                "weighted_score": relation["__metrics__"],
                "description": relation_content["description"],
            }
        )

    normalize_list(list_objs=entities, key="weighted_score")
    normalize_list(list_objs=relationships, key="weighted_score")
    normalize_list(list_objs=chunks, key="weighted_score")

    return chunks, entities, relationships


async def fuse_kg_text(
    query,
    chunks_vdb,
    text_chunks_db,
    knowledge_graph,
    entity_vdb,
    relationship_vdb,
    use_model_func,
    hyperparams=None,
    # ── Image modality (optional) ──
    images_vdb=None,
    image_text_vdb=None,
    query_image_path=None,
):
    """
    Fuse knowledge graph and text retrieval with cross-weighting.
    Supports optional image modality for cross-modal fusion.

    Returns ranked chunks, entities, relationships, and optionally images.
    """

    if hyperparams is None:
        hyperparams = {}

    max_token_size = hyperparams.get("max_token_size", 4000)
    ent_to_chunk_wt = hyperparams.get("ent_to_chunk_wt", 0.5)
    rel_to_chunk_wt = hyperparams.get("rel_to_chunk_wt", 0.5)
    chunk_to_ent_wt = hyperparams.get("chunk_to_ent_wt", 0.5)
    chunk_to_rel_wt = hyperparams.get("chunk_to_rel_wt", 0.5)
    # Cross-modal weights
    lambda_image_source = hyperparams.get("lambda_image_source", 0.5)
    lambda_cross_identity = hyperparams.get("lambda_cross_identity", 1.0)
    lambda_cross_depicts = hyperparams.get("lambda_cross_depicts", 0.5)
    lambda_text_source = hyperparams.get("lambda_text_source", 0.5)
    etop_k = hyperparams.get("etop_k", 60)
    top_k = hyperparams.get("top_k", 60)
    img_top_k = hyperparams.get("img_top_k", 60)

    iter_times = 1

    try:
        keywords_relationships = await _extract_keywords_relationships(
            query, use_model_func
        )
    except Exception:
        logger.error("Failed to extract keywords and relationships!")
        return "\nFailed to extract keywords and relationships!\n"

    chunks_result, entities_result, relations_results = await _parallel_retrieval(
        query,
        keywords_relationships,
        chunks_vdb,
        entity_vdb,
        relationship_vdb,
        top_k,
        etop_k,
    )

    # ── Image retrieval (parallel) ──
    images_result = []
    image_text_result = []
    if images_vdb is not None:
        try:
            images_result = await images_vdb.query(query, top_k=img_top_k)
        except Exception:
            logger.warning("Image vector DB query failed, continuing without images.")
        if query_image_path and hasattr(images_vdb, "query_by_image"):
            try:
                query_image_results = await images_vdb.query_by_image(
                    query_image_path, top_k=img_top_k
                )
                images_result.extend(query_image_results)
            except Exception:
                logger.warning("Image-to-image vector DB query failed, continuing without it.")
    if image_text_vdb is not None:
        try:
            image_text_result = await image_text_vdb.query(query, top_k=img_top_k)
        except Exception:
            logger.warning("Image text VDB query failed, continuing without images.")

    chunks, entities, relationships = await process_retrieval_results(
        chunks_result, entities_result, relations_results, knowledge_graph
    )

    text_dict, entities_dict, relationships_dict = _initialize_retrieval_dicts(
        chunks_result, entities_result, relations_results
    )

    # ── Initialize images_dict ──
    images_dict = {}
    has_images = images_vdb is not None and len(images_result) > 0
    if has_images:
        for img in images_result[:15]:
            images_dict[img["id"]] = {
                "id": img["id"],
                "weighted_score": img.get("__metrics__", img.get("distance", 0.0)),
            }
    if image_text_result:
        for img in image_text_result[:15]:
            score = img.get("__metrics__", img.get("distance", 0.0))
            if img["id"] not in images_dict:
                images_dict[img["id"]] = {"id": img["id"], "weighted_score": score}
            else:
                images_dict[img["id"]]["weighted_score"] += score
    has_images = bool(images_dict)
    if has_images:
        normalize_list(images_dict.values(), "weighted_score")

    for i in range(iter_times):
        # Existing 4 directions
        (
            c_to_e_contrib,
            c_to_r_contrib,
            e_to_c_contrib,
            r_to_c_contrib,
        ) = await asyncio.gather(
            _compute_chunk_to_entity_contributions(
                text_dict, knowledge_graph, entity_vdb, chunk_to_ent_wt
            ),
            _compute_chunk_to_relation_contributions(
                text_dict, knowledge_graph, relationship_vdb, chunk_to_rel_wt
            ),
            _compute_entity_to_chunk_contributions(
                entities_dict, knowledge_graph, ent_to_chunk_wt
            ),
            _compute_relation_to_chunk_contributions(
                relationships_dict, knowledge_graph, rel_to_chunk_wt
            ),
        )

        # ── Cross-modal image → entity contributions ──
        img_to_ent_contrib = {}
        img_to_chunk_contrib = {}
        ent_to_img_contrib = {}
        chunk_to_img_contrib = {}
        rel_to_img_contrib = {}
        img_to_rel_contrib = {}

        if has_images:
            img_to_ent_contrib = await _compute_image_to_entity_contributions(
                images_dict, knowledge_graph, entity_vdb, lambda_image_source
            )
            img_to_rel_contrib = await _compute_image_to_relation_contributions(
                images_dict, knowledge_graph, relationship_vdb, lambda_image_source
            )
            img_to_chunk_contrib = await _compute_image_to_chunk_contributions(
                images_dict, knowledge_graph, lambda_image_source
            )
            ent_to_img_contrib = await _compute_entity_to_image_contributions(
                entities_dict, knowledge_graph, lambda_cross_depicts
            )
            rel_to_img_contrib = await _compute_relation_to_image_contributions(
                relationships_dict, knowledge_graph, lambda_cross_depicts
            )
            chunk_to_img_contrib = await _compute_chunk_to_image_contributions(
                text_dict, knowledge_graph, lambda_text_source
            )

        _apply_contributions_to_dicts(
            text_dict,
            entities_dict,
            relationships_dict,
            c_to_e_contrib,
            c_to_r_contrib,
            e_to_c_contrib,
            r_to_c_contrib,
            # Cross-modal
            img_to_ent_contrib=img_to_ent_contrib,
            img_to_chunk_contrib=img_to_chunk_contrib,
            ent_to_img_contrib=ent_to_img_contrib,
            chunk_to_img_contrib=chunk_to_img_contrib,
            rel_to_img_contrib=rel_to_img_contrib,
            img_to_rel_contrib=img_to_rel_contrib,
            images_dict=images_dict,
        )

        text_dict, entities_dict, relationships_dict = _sort_and_truncate_dicts(
            text_dict, entities_dict, relationships_dict, 50
        )
        if images_dict:
            images_dict = dict(
                sorted(images_dict.items(), key=lambda x: x[1]["weighted_score"], reverse=True)[:30]
            )

        normalize_list(text_dict.values(), "weighted_score")
        normalize_list(entities_dict.values(), "weighted_score")
        normalize_list(relationships_dict.values(), "weighted_score")
        if images_dict:
            normalize_list(images_dict.values(), "weighted_score")

        ranked_chunks, ranked_entities, ranked_relations = await _finalize_results(
            text_dict,
            entities_dict,
            relationships_dict,
            text_chunks_db,
            knowledge_graph,
        )

        # ── Build ranked images ──
        ranked_images = list(images_dict.values()) if images_dict else []

        return (
            ranked_chunks,
            ranked_entities,
            ranked_relations,
            chunks,
            entities,
            relationships,
            ranked_images,
        )


# ──────────────────────────────────────────────
# Cross-Modal Score Propagation Functions
# ──────────────────────────────────────────────

def _is_cross_modal_edge(edge):
    keywords = (edge or {}).get("keywords", "").lower()
    return "identity equivalence" in keywords or "likely depicts" in keywords


async def _compute_image_to_entity_contributions(
    images_dict, knowledge_graph, entity_vdb, img_to_ent_wt
):
    """Image → Entity: image chunk entities plus linked text entities."""
    contrib = {}
    for img_id, img_item in images_dict.items():
        score = img_item["weighted_score"]
        try:
            ents, _ = await knowledge_graph.find_by_chunk(img_id)
            for entity_name, _ in ents:
                if entity_name.startswith("IMAGE:"):
                    continue
                eid = await entity_vdb.get_entity_id(name=entity_name)
                if eid:
                    c = score * img_to_ent_wt
                    if eid not in contrib:
                        contrib[eid] = {"weighted_score": c, "entity_name": entity_name}
                    else:
                        contrib[eid]["weighted_score"] += c
                for neighbor in await knowledge_graph.get_neighbors(entity_name):
                    if neighbor.startswith("IMAGE:") or neighbor.startswith("VISUAL_ENTITY:"):
                        continue
                    edge = await knowledge_graph.get_edge(entity_name, neighbor)
                    if not edge or not _is_cross_modal_edge(edge):
                        continue
                    linked_eid = await entity_vdb.get_entity_id(name=neighbor)
                    if linked_eid:
                        c = score * img_to_ent_wt
                        if linked_eid not in contrib:
                            contrib[linked_eid] = {
                                "weighted_score": c,
                                "entity_name": neighbor,
                            }
                        else:
                            contrib[linked_eid]["weighted_score"] += c
        except Exception:
            continue
    return contrib


async def _compute_image_to_relation_contributions(
    images_dict, knowledge_graph, relationship_vdb, img_to_rel_wt
):
    """Image → Relation: find_by_chunk(img_chunk_id) → relations in this image."""
    contrib = {}
    for img_id, img_item in images_dict.items():
        score = img_item["weighted_score"]
        try:
            _, rels = await knowledge_graph.find_by_chunk(img_id)
            for src, tgt, _ in rels:
                rid = await relationship_vdb.get_rel_id(src=src, tgt=tgt)
                if rid is None:
                    rid = await relationship_vdb.get_rel_id(src=tgt, tgt=src)
                if rid:
                    c = score * img_to_rel_wt
                    if rid not in contrib:
                        contrib[rid] = {
                            "weighted_score": c,
                            "src_name": src,
                            "tgt_name": tgt,
                        }
                    else:
                        contrib[rid]["weighted_score"] += c
        except Exception:
            continue
    return contrib


async def _compute_image_to_chunk_contributions(
    images_dict, knowledge_graph, img_to_chunk_wt
):
    """Image → Chunk: through visual entities linked to text entities."""
    contrib = {}
    for img_id, img_item in images_dict.items():
        score = img_item["weighted_score"]
        seen_chunks = set()
        try:
            ents, _ = await knowledge_graph.find_by_chunk(img_id)
            for entity_name, entity_attr in ents:
                node = await knowledge_graph.get_node(entity_name)
                if node is None:
                    continue
                candidate_nodes = [node]
                for neighbor in await knowledge_graph.get_neighbors(entity_name):
                    edge = await knowledge_graph.get_edge(entity_name, neighbor)
                    if edge and _is_cross_modal_edge(edge):
                        linked_node = await knowledge_graph.get_node(neighbor)
                        if linked_node:
                            candidate_nodes.append(linked_node)
                for candidate in candidate_nodes:
                    source_ids = candidate.get("source_id", "").split("<SEP>")
                    for sid in source_ids:
                        sid = sid.strip()
                        if sid and sid.startswith("chunk-") and sid not in seen_chunks:
                            seen_chunks.add(sid)
                            contrib[sid] = contrib.get(sid, 0.0) + score * img_to_chunk_wt
        except Exception:
            continue
    return contrib


async def _compute_entity_to_image_contributions(
    entities_dict, knowledge_graph, ent_to_img_wt
):
    """Entity → Image: entity source IDs plus cross-modal edges."""
    contrib = {}
    for eid, ent_item in entities_dict.items():
        score = ent_item["weighted_score"]
        entity_name = ent_item["entity_name"]
        try:
            node = await knowledge_graph.get_node(entity_name)
            if node is None:
                continue
            source_ids = node.get("source_id", "").split("<SEP>")
            for sid in source_ids:
                sid = sid.strip()
                if sid and sid.startswith("img-"):
                    contrib[sid] = contrib.get(sid, 0.0) + score * ent_to_img_wt
            for neighbor in await knowledge_graph.get_neighbors(entity_name):
                edge = await knowledge_graph.get_edge(entity_name, neighbor)
                if not edge or not _is_cross_modal_edge(edge):
                    continue
                for sid in edge.get("source_id", "").split("<SEP>"):
                    sid = sid.strip()
                    if sid.startswith("img-"):
                        contrib[sid] = contrib.get(sid, 0.0) + score * ent_to_img_wt
        except Exception:
            continue
    return contrib


async def _compute_relation_to_image_contributions(
    relationships_dict, knowledge_graph, rel_to_img_wt
):
    """Relation → Image: relation's source_id may contain image chunk IDs."""
    contrib = {}
    for rid, rel_item in relationships_dict.items():
        score = rel_item["weighted_score"]
        src, tgt = rel_item["src_name"], rel_item["tgt_name"]
        try:
            edge = await knowledge_graph.get_edge(src, tgt)
            if edge is None:
                continue
            source_ids = edge.get("source_id", "").split("<SEP>")
            for sid in source_ids:
                sid = sid.strip()
                if sid and sid.startswith("img-"):
                    contrib[sid] = contrib.get(sid, 0.0) + score * rel_to_img_wt
        except Exception:
            continue
    return contrib


async def _compute_chunk_to_image_contributions(
    text_dict, knowledge_graph, chunk_to_img_wt
):
    """Chunk → Image: chunk entities and their cross-modal image links."""
    contrib = {}
    for cid, chunk_item in text_dict.items():
        score = chunk_item["weighted_score"]
        try:
            ents, _ = await knowledge_graph.find_by_chunk(cid)
            seen_imgs = set()
            for entity_name, entity_attr in ents:
                source_ids = entity_attr.get("source_id", "").split("<SEP>")
                for sid in source_ids:
                    sid = sid.strip()
                    if sid and sid.startswith("img-") and sid not in seen_imgs:
                        seen_imgs.add(sid)
                        contrib[sid] = contrib.get(sid, 0.0) + score * chunk_to_img_wt
                for neighbor in await knowledge_graph.get_neighbors(entity_name):
                    edge = await knowledge_graph.get_edge(entity_name, neighbor)
                    if not edge or not _is_cross_modal_edge(edge):
                        continue
                    for sid in edge.get("source_id", "").split("<SEP>"):
                        sid = sid.strip()
                        if sid.startswith("img-") and sid not in seen_imgs:
                            seen_imgs.add(sid)
                            contrib[sid] = contrib.get(sid, 0.0) + score * chunk_to_img_wt
        except Exception:
            continue
    return contrib
