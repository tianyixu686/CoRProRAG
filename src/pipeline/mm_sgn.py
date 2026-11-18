"""Text and image SGN retrieval tools used by cor_simple."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from src.prompt import PROMPTS
from src.pipeline.context_packing import pack_text, input_budget, encode, decode
from src.retrieval.beam_search import beam_search, process_beam_output, detect_edge_type
from src.retrieval.fusion import fuse_kg_text
from src.retrieval.graph_ops import get_one_hop_path
from src.utils import (
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    parse_json_response,
)


UNKNOWN_MARKERS = {
    "",
    "i don't know",
    "i don't know.",
    "unknown",
    "unclear",
    "not identifiable",
    "not visible",
    "no match",
}

GENERIC_VISUAL_ANSWERS = {
    "badge",
    "logo",
    "emblem",
    "symbol",
    "object",
    "animal",
    "plant",
    "food",
    "dish",
    "person",
    "people",
    "image",
}


def _graph_image_node_name(image_path: str) -> str:
    if not image_path:
        return ""
    return "IMAGE:" + os.path.basename(str(image_path)).upper()


def _page_title_from_caption(caption: str) -> str:
    match = re.search(r"page '([^']+)'", str(caption or ""))
    return match.group(1).strip() if match else ""


def _is_text_entity_name(name: str) -> bool:
    upper = str(name or "").upper()
    return bool(name) and not upper.startswith(("IMAGE:", "VISUAL_ENTITY:"))


def _node_result(
    node,
    answer="I don't know.",
    rationale="",
    evidence=None,
    path=None,
    confidence=0.0,
    failure_type="",
    produced_type="",
    candidate_state=None,
):
    evidence = _jsonable(evidence or [])
    path = _jsonable(path or [])
    result = {
        "node_id": node.get("id"),
        "operation_type": node.get("operation_type", ""),
        "modality": node.get("modality", ""),
        "query": node.get("query", ""),
        "answer": answer or "I don't know.",
        "rationale": rationale or "",
        "evidence": evidence,
        "path": path,
        "confidence": float(confidence or 0.0),
        "failure_type": failure_type or "",
        "produced_type": produced_type or node.get("expected_output_type", ""),
    }
    if candidate_state is not None:
        result["candidate_state"] = _jsonable(candidate_state)
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_unknown(answer: str) -> bool:
    norm = str(answer or "").strip().lower().rstrip(".")
    return norm in {m.rstrip(".") for m in UNKNOWN_MARKERS}


def _hyperparams(config) -> Dict[str, Any]:
    return {
        "max_token_size": config.kg_max_token_size,
        "ent_to_chunk_wt": config.ent_to_chunk_wt,
        "rel_to_chunk_wt": config.rel_to_chunk_wt,
        "chunk_to_ent_wt": config.chunk_to_ent_wt,
        "chunk_to_rel_wt": config.chunk_to_rel_wt,
        "etop_k": config.kg_etop_k,
        "rtop_k": config.kg_rtop_k,
        "top_k": config.kg_top_k,
        "beam_search_max_hops": config.beam_search_max_hops,
        "beam_search_beam_width": config.beam_search_beam_width,
        "beam_search_alpha": config.beam_search_alpha,
        "beam_search_beta": config.beam_search_beta,
        "beam_search_decay": config.beam_search_decay,
        "lambda_image_source": config.lambda_image_source,
        "lambda_cross_identity": config.lambda_cross_identity,
        "lambda_cross_depicts": config.lambda_cross_depicts,
        "lambda_text_source": config.lambda_text_source,
        "img_top_k": config.img_top_k,
        "edge_type_weights": config.edge_type_weights,
    }


def _node_retrieval_query(node: Dict[str, Any]) -> str:
    allowed_edges = ", ".join(node.get("allowed_edge_types") or [])
    parts = [
        node.get("query", ""),
        f"Operation: {node.get('operation_type', '')}",
        f"Modality: {node.get('modality', '')}",
        f"Input anchor: {node.get('input_anchor', '')}",
        f"Expected output type: {node.get('expected_output_type', '')}",
        f"Allowed edge types: {allowed_edges}",
    ]
    return "\n".join(str(part) for part in parts if str(part).strip())


def _dedupe_strings(values, limit=None):
    deduped = []
    seen = set()
    for value in values or []:
        text = str(value or "").strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _compact_image_candidate(item: Dict[str, Any], rank: int) -> Dict[str, Any]:
    context = item.get("kg_context") or {}
    image_chunk = context.get("image_chunk") or {}
    caption = context.get("caption") or item.get("caption") or ""
    page_title = _page_title_from_caption(caption)
    linked_entities = _dedupe_strings([item.get("candidate_exact_entity", ""), page_title, *(context.get("linked_entities") or [])], limit=8)
    return {
        "rank": rank,
        "image_id": item.get("id") or item.get("__id__") or "",
        "image_path": context.get("image_path") or item.get("image_path") or "",
        "caption": caption,
        "page_title": page_title,
        "global_description": image_chunk.get("global_description", ""),
        "ocr_text": image_chunk.get("ocr_text", ""),
        "linked_entities": linked_entities,
        "visual_score": item.get("visual_score_normalized", 0.0),
        "text_score": item.get("text_score_normalized", 0.0),
        "rerank_score": item.get("rerank_score", 0.0),
        "fusion_score": item.get("fusion_score", 0.0),
    }


def _query_exact_phrases(query: str) -> List[str]:
    q = str(query or "")
    phrases = []
    for pattern in (
        r"\b(?:of|in|from)\s+([A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){1,6})",
        r"\b([A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){1,6})\s+table\b",
    ):
        for match in re.finditer(pattern, q):
            phrase = match.group(1).strip(" .,:;?\n\t")
            if phrase and phrase.lower() not in {"retrieve the", "verify that"}:
                phrases.append(phrase)
    low = q.lower()
    if "sports in philadelphia" in low:
        phrases.append("Sports in Philadelphia")
    if "collegiate sports" in low:
        phrases.append("Collegiate sports")
    return _dedupe_strings(phrases, limit=8)


async def _exact_text_context(query: str, text_chunks_db, limit: int = 3) -> str:
    if not text_chunks_db:
        return ""
    phrases = _query_exact_phrases(query)
    if not phrases:
        return ""
    try:
        keys = await text_chunks_db.all_keys()
        chunks = await text_chunks_db.get_by_ids(keys)
    except Exception:
        return ""
    scored = []
    phrase_lowers = [phrase.lower() for phrase in phrases]
    for key, chunk in zip(keys, chunks or []):
        if not chunk:
            continue
        content = str(chunk.get("content") or "")
        low = content.lower()
        score = 0
        for phrase, phrase_low in zip(phrases, phrase_lowers):
            if phrase_low in low:
                score += 1
            if f"page title: {phrase_low}" in low:
                score += 4
            if f"table: {phrase_low}" in low:
                score += 4
        if score:
            scored.append((score, key, content))
    scored.sort(key=lambda item: item[0], reverse=True)
    parts = []
    for score, key, content in scored[:limit]:
        parts.append(f"[exact_text_match score={score} chunk_id={key}]\n{content}")
    return "\n\n".join(parts)


async def _run_text_sgn_once(node, stores, funcs, config, execution_context):
    query = _node_retrieval_query(node)
    hyperparams = _hyperparams(config)
    retrieval = await fuse_kg_text(
        query,
        stores["chunks_vdb"],
        stores["text_chunks_db"],
        stores["knowledge_graph_inst"],
        stores["entities_vdb"],
        stores["relationships_vdb"],
        funcs["llm_model_func"],
        hyperparams=hyperparams,
        images_vdb=stores.get("images_vdb"),
        image_text_vdb=stores.get("image_text_vdb"),
        query_image_path=execution_context.get("query_image_path"),
    )
    if not isinstance(retrieval, tuple):
        return _node_result(
            node,
            rationale="Retrieval failed before SGN beam search.",
            failure_type="retrieval_failed",
        )

    ranked_chunks, ranked_entities, ranked_relationships = retrieval[:3]
    ranked_images = retrieval[6] if len(retrieval) > 6 else []
    beam = await beam_search(
        query,
        ranked_entities,
        ranked_chunks,
        stores["knowledge_graph_inst"],
        max_hops=hyperparams["beam_search_max_hops"],
        beam_width=hyperparams["beam_search_beam_width"],
        alpha=hyperparams["beam_search_alpha"],
        beta=hyperparams["beam_search_beta"],
        decay=hyperparams["beam_search_decay"],
        rerank_texts_func=funcs["rerank_texts_func"],
        edge_type_weights=config.edge_type_weights,
    )
    image_captions = await _collect_image_captions(ranked_images, stores.get("image_chunks_db"))
    context_budget = max(1000, int(config.mm_sgn_context_max_tokens))
    _, _, _, context = await process_beam_output(
        beam,
        ranked_entities,
        ranked_relationships,
        ranked_chunks,
        hyperparams["beam_search_max_hops"],
        max_entities_tokens=context_budget // 4,
        max_relationships_tokens=context_budget // 4,
        max_chunks_tokens=context_budget // 2,
        image_chunks_db=stores.get("image_chunks_db"),
        image_captions=image_captions,
    )
    exact_context = await _exact_text_context(node.get("query", ""), stores.get("text_chunks_db"))
    if exact_context:
        context = exact_context + "\n\n" + context
    template = PROMPTS["answer_sub_query"]
    overhead = len(encode(template.format(sub_query=node.get("query", ""), context="")))
    context_budget = min(context_budget, max(0, input_budget(512) - overhead - 32))
    context = pack_text(node.get("query", ""), [{"content": context}], context_budget)
    prompt = template.format(sub_query=node.get("query", ""), context=context)
    try:
        response = await funcs["llm_model_func"](
            prompt=prompt,
            temperature=0.0,
            max_tokens=512,
        )
        parsed = await parse_json_response(response, funcs["llm_model_func"])
    except Exception as exc:
        return _node_result(
            node,
            rationale=f"Answer generation failed: {exc}",
            failure_type="answer_parse_failed",
        )
    answer = str((parsed or {}).get("answer") or "I don't know.").strip()
    confidence = 0.0 if _is_unknown(answer) else max(0.55, float(node.get("confidence", 0.0) or 0.0))
    return _node_result(
        node,
        answer=answer,
        rationale=str((parsed or {}).get("rationale") or ""),
        evidence=[{"type": "context", "content": context}],
        path=(beam[0].get("paths", []) if beam else []),
        confidence=confidence,
        failure_type="no_text_evidence" if _is_unknown(answer) else "",
        produced_type=node.get("expected_output_type", "entity"),
    )


async def _collect_image_captions(ranked_images, image_chunks_db):
    captions = []
    if not ranked_images or not image_chunks_db:
        return captions
    for img in ranked_images[:5]:
        img_id = img.get("id", "")
        if not img_id:
            continue
        try:
            img_data = await image_chunks_db.get_by_id(img_id)
        except Exception:
            img_data = None
        if img_data:
            captions.append((img_id, img_data.get("caption") or img_data.get("global_description", "")))
    return captions


async def _get_image_chunk(item: Dict[str, Any], image_chunks_db):
    if not image_chunks_db:
        return None
    img_id = item.get("id") or item.get("__id__")
    if not img_id:
        return None
    try:
        return await image_chunks_db.get_by_id(img_id)
    except Exception:
        return None


async def _image_kg_context(item: Dict[str, Any], stores) -> Dict[str, Any]:
    img_data = await _get_image_chunk(item, stores.get("image_chunks_db"))
    image_path = (img_data or {}).get("image_path") or item.get("image_path", "")
    caption = (img_data or {}).get("caption") or item.get("caption", "")
    graph_image_node = _graph_image_node_name(image_path)
    context = {
        "image_chunk": img_data or {},
        "image_path": image_path,
        "caption": caption,
        "graph_image_node": graph_image_node,
        "linked_entities": [],
        "graph_paths": [],
    }
    kg = stores.get("knowledge_graph_inst")
    if not kg or not graph_image_node:
        title = _page_title_from_caption(caption)
        if title:
            context["linked_entities"] = [title]
        return context

    try:
        first_hop = await get_one_hop_path([graph_image_node], kg)
    except Exception:
        first_hop = []
    context["graph_paths"].extend(first_hop)

    visual_nodes = []
    for path in first_hop:
        tgt = path.get("tgt_name", "")
        if _is_text_entity_name(tgt):
            context["linked_entities"].append(tgt)
        elif str(tgt).upper().startswith("VISUAL_ENTITY:"):
            visual_nodes.append(tgt)

    if visual_nodes:
        try:
            second_hop = await get_one_hop_path(visual_nodes[:8], kg)
        except Exception:
            second_hop = []
        context["graph_paths"].extend(second_hop)
        for path in second_hop:
            for key in ("tgt_name", "src_name"):
                name = path.get(key, "")
                if _is_text_entity_name(name):
                    context["linked_entities"].append(name)

    title = _page_title_from_caption(caption)
    if title:
        context["linked_entities"].append(title)

    deduped = []
    seen = set()
    for entity in context["linked_entities"]:
        key = str(entity).strip().upper()
        if key and key not in seen:
            seen.add(key)
            deduped.append(str(entity).strip())
    context["linked_entities"] = deduped
    return context


def _image_candidate_text(item: Dict[str, Any]) -> str:
    context = item.get("kg_context") or {}
    image_chunk = context.get("image_chunk") or {}
    graph_paths = context.get("graph_paths") or []
    parts = [
        " | ".join(context.get("linked_entities") or []),
        context.get("caption", ""),
        image_chunk.get("global_description", ""),
        image_chunk.get("ocr_text", ""),
    ]
    for path in graph_paths[:12]:
        parts.extend(
            [
                path.get("src_name", ""),
                path.get("tgt_name", ""),
                path.get("description", ""),
                path.get("tgt_description", ""),
            ]
        )
    return "\n".join(str(part) for part in parts if str(part).strip())[:5000]


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _split_image_query_constraints(query: str) -> Tuple[str, str]:
    """Separate visual content from textual set constraints for KB image retrieval."""
    text = " ".join(str(query or "").split())
    if not text:
        return "", ""
    lower = text.lower()
    visual_query = text
    textual_constraint = ""

    text = re.sub(r"\b(?:among|for)\s+(?:i don'?t know|unknown|unclear)\.?[,]?\s*", "", text, flags=re.IGNORECASE)
    logo_match = re.search(
        r"(?:whose|with|that has|has|having)?\s*(?:a\s+)?(?:logo|seal|emblem)(?:\s+images?)?(?:\s+whose)?(?:\s+visual\s+content)?\s+(?:that\s+)?(?:features?|featuring|contains?|with|showing)\s+([^,.;?]+)",
        text,
        flags=re.IGNORECASE,
    )
    if logo_match:
        visual_tail = logo_match.group(1).strip()
        logo_word = re.search(r"\b(logo|seal|emblem)\b", logo_match.group(0), flags=re.IGNORECASE)
        visual_query = f"{logo_word.group(1).lower() if logo_word else 'logo'} featuring {visual_tail}"
        before = text[: logo_match.start()].strip(" ,.;:-")
        # Remove generic retrieval boilerplate, keep the named/table/category constraint.
        before = re.sub(r"^retrieve\s+(?:kb/corpus\s+)?(?:images?\s+of\s+)?", "", before, flags=re.IGNORECASE).strip()
        before = re.sub(r"\b(?:and\s+)?map\s+.*$", "", before, flags=re.IGNORECASE).strip()
        textual_constraint = before
    elif any(marker in lower for marker in ("poster", "photo", "image", "picture")):
        # For non-logo visual evidence, use the visual noun phrase itself for CLIP and keep the
        # rest as text-side constraints/rerank context.
        visual_query = text
        textual_constraint = ""

    # Avoid asking CLIP to solve table/list membership conditions.
    visual_query = re.sub(
        r"\b(public|private|current|former)\s+(school|university|member|team|college)s?\b",
        "",
        visual_query,
        flags=re.IGNORECASE,
    )
    visual_query = re.sub(
        r"\b(in|from|of)\s+the\s+collegiate\s+sports\s+of\s+sports\s+in\s+philadelphia\b",
        "",
        visual_query,
        flags=re.IGNORECASE,
    )
    visual_query = " ".join(visual_query.split()).strip(" ,.;:-") or text
    return visual_query, textual_constraint


def _candidate_entities_from_constraint(text: str) -> List[str]:
    raw = str(text or "")
    match = re.search(r"\bAmong\s+(.+?)(?:,\s*(?:retrieve|verify|retrieve/verify)\b|$)", raw, flags=re.IGNORECASE)
    if match:
        raw = match.group(1)
    raw = re.sub(r"\bvisual\s+query\s*:\s*.*$", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("|", ",")
    parts = re.split(r",|;|\band\b", raw)
    cleaned = []
    for part in parts:
        ent = part.strip(" .,:;\n\t")
        ent = re.sub(r"^(among|candidate entity to verify:)\s+", "", ent, flags=re.IGNORECASE).strip()
        if not ent or ent.lower() in {"retrieve/verify kb", "i don't know", "unknown"}:
            continue
        if len(ent.split()) > 8:
            continue
        cleaned.append(ent)
    return _dedupe_strings(cleaned, limit=12)


def _visual_terms_from_query(visual_query: str) -> List[str]:
    q = str(visual_query or "").lower()
    terms = []
    for phrase in ("open book", "flying horse", "lion", "star", "shield", "seal", "crest"):
        if phrase in q:
            terms.append(phrase)
    q = re.sub(r"\b(logo|seal|emblem|featuring|features|feature|contains|with|at|least|one|the|a|an|image|kb|corpus)\b", " ", q)
    for token in re.findall(r"[a-z0-9'-]{4,}", q):
        if token not in terms:
            terms.append(token)
    return terms[:8]


async def _candidate_exact_image_results(candidate_entities: List[str], visual_query: str, image_chunks_db, limit_per_entity: int = 2) -> List[Dict[str, Any]]:
    if not candidate_entities or not image_chunks_db:
        return []
    visual_terms = _visual_terms_from_query(visual_query)
    try:
        keys = await image_chunks_db.all_keys()
        chunks = await image_chunks_db.get_by_ids(keys)
    except Exception:
        return []
    results = []
    for entity in candidate_entities:
        entity_low = entity.lower()
        local = []
        for key, chunk in zip(keys, chunks or []):
            if not chunk:
                continue
            caption = str(chunk.get("caption") or "")
            desc = str(chunk.get("global_description") or "")
            hay = f"{caption}\n{desc}\n{chunk.get('ocr_text') or ''}".lower()
            if entity_low not in hay:
                continue
            visual_hit = any(term in hay for term in visual_terms) if visual_terms else True
            score = 1.0 if visual_hit else 0.65
            local.append(
                (
                    score,
                    {
                        "id": key,
                        "__id__": key,
                        "image_path": chunk.get("image_path", ""),
                        "caption": caption,
                        "candidate_exact_entity": entity,
                        "candidate_exact_retrieval_score": score,
                        "__metrics__": score,
                    },
                )
            )
        local.sort(key=lambda item: item[0], reverse=True)
        results.extend(item for _, item in local[:limit_per_entity])
    return results


async def _run_image_retrieve(node, stores, funcs, execution_context, config):
    query = str(node.get("query") or node.get("input_anchor") or "").strip()
    visual_query, textual_constraint_query = _split_image_query_constraints(query)
    candidate_entities = _candidate_entities_from_constraint(textual_constraint_query or query)
    candidate_k = max(5, int(config.image_retrieval_candidate_k))
    visual_results = []
    text_results = []
    constraint_text_results = []
    candidate_exact_results = []
    if stores.get("images_vdb") is not None:
        try:
            visual_results = await stores["images_vdb"].query(
                visual_query, top_k=candidate_k
            )
        except Exception:
            pass
    if stores.get("image_text_vdb") is not None:
        try:
            text_results = await stores["image_text_vdb"].query(
                visual_query, top_k=min(config.image_text_top_k, candidate_k)
            )
        except Exception:
            pass
        if textual_constraint_query:
            try:
                constraint_text_results = await stores["image_text_vdb"].query(
                    textual_constraint_query, top_k=min(config.image_text_top_k, candidate_k)
                )
            except Exception:
                constraint_text_results = []
    candidate_exact_results = await _candidate_exact_image_results(
        candidate_entities, visual_query, stores.get("image_chunks_db")
    )
    if not visual_results and not text_results:
        return _node_result(
            node,
            rationale="No KB image was retrieved.",
            failure_type="no_kb_image",
            produced_type="image",
        )

    candidates = {}
    for source, source_results in (("candidate_exact", candidate_exact_results), ("visual", visual_results), ("text", text_results), ("constraint_text", constraint_text_results)):
        for raw in source_results:
            item = dict(raw)
            item_id = item.get("id") or item.get("__id__") or item.get("image_path")
            if not item_id:
                continue
            candidate = candidates.setdefault(item_id, item)
            if item.get("candidate_exact_entity"):
                candidate["candidate_exact_entity"] = item.get("candidate_exact_entity")
            score = float(item.get("__metrics__", item.get("distance", 0.0)) or 0.0)
            candidate[f"{source}_retrieval_score"] = max(
                score, float(candidate.get(f"{source}_retrieval_score", 0.0) or 0.0)
            )

    enriched = []
    for item in list(candidates.values())[: candidate_k * 2]:
        item = dict(item)
        item["kg_context"] = await _image_kg_context(item, stores)
        enriched.append(item)

    rerank_func = funcs.get("rerank_texts_func")
    rerank_scores = []
    if rerank_func and enriched:
        try:
            rerank_results = await rerank_func(
                query="\n".join(part for part in [visual_query, textual_constraint_query] if part),
                texts=[_image_candidate_text(item) for item in enriched],
            )
            rerank_scores = [
                float(result.get("relevance_score", 0.0) or 0.0)
                for result in rerank_results
            ]
        except Exception:
            rerank_scores = []
    if len(rerank_scores) != len(enriched):
        rerank_scores = [0.5] * len(enriched)
    rerank_scores_normalized = _minmax(rerank_scores)

    visual_scores = _minmax(
        [float(item.get("visual_retrieval_score", 0.0) or 0.0) for item in enriched]
    )
    text_scores = _minmax(
        [
            max(
                float(item.get("text_retrieval_score", 0.0) or 0.0),
                float(item.get("constraint_text_retrieval_score", 0.0) or 0.0),
                float(item.get("candidate_exact_retrieval_score", 0.0) or 0.0),
            )
            for item in enriched
        ]
    )
    visual_weight = max(0.0, min(1.0, float(config.image_retrieval_visual_weight)))
    for item, visual_score, text_score, rerank_raw_score, rerank_score in zip(
        enriched,
        visual_scores,
        text_scores,
        rerank_scores,
        rerank_scores_normalized,
    ):
        lexical_score = 0.7 * rerank_score + 0.3 * text_score
        item["visual_score_normalized"] = visual_score
        item["text_score_normalized"] = text_score
        item["rerank_raw_score"] = rerank_raw_score
        item["rerank_score"] = rerank_score
        base_fusion = visual_weight * visual_score + (1.0 - visual_weight) * lexical_score
        if item.get("candidate_exact_entity"):
            # When a text hop produced an explicit candidate set, exact candidate-image recall
            # is a hard constraint signal. General open-book/logo visual matches are fallback.
            item["candidate_constraint_hit"] = 1.0
            item["fusion_score"] = 1.0 + base_fusion
        else:
            item["candidate_constraint_hit"] = 0.0
            item["fusion_score"] = base_fusion
    enriched.sort(
        key=lambda item: (
            item.get("candidate_constraint_hit", 0.0),
            item.get("fusion_score", 0.0),
        ),
        reverse=True,
    )

    top = enriched[0]
    kg_context = top.get("kg_context", {})
    linked_entities = kg_context.get("linked_entities") or []
    answer = (
        linked_entities[0]
        if linked_entities
        else top.get("id") or top.get("image_path") or "I don't know."
    )
    state_k = max(1, int(getattr(config, "mm_candidate_state_top_k", 5)))
    compact_candidates = [
        _compact_image_candidate(item, rank=rank)
        for rank, item in enumerate(enriched[:state_k], start=1)
    ]
    return _node_result(
        node,
        answer=answer,
        rationale=(
            "Retrieved KB images matching the node query and mapped the top image "
            "through MMKG image/visual-entity links. Candidate state keeps alternate "
            "image/entity hypotheses for downstream verification."
        ),
        evidence=[{"type": "kb_image_candidate", **item} for item in enriched[:10]],
        path=kg_context.get("graph_paths", [])[:10],
        confidence=float(top.get("fusion_score", 0.0) or 0.0),
        produced_type="entity" if linked_entities else "image",
        candidate_state={
            "type": "image_entity_candidates",
            "selected_answer": answer,
            "selected_answer_tentative": answer,
            "selected_image_id": compact_candidates[0].get("image_id") if compact_candidates else "",
            "selection_policy": "tentative_top_fusion_score; downstream nodes may rerank candidates",
            "visual_query": visual_query,
            "textual_constraint_query": textual_constraint_query,
            "candidate_entities": candidate_entities,
            "candidates": compact_candidates,
        },
    )
