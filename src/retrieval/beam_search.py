"""Beam search for multi-hop reasoning on knowledge graphs."""

import asyncio
import random
import re

from src.utils import (
    truncate_list_by_token_size_async,
    truncate_by_token_size_async,
    normalize_list,
)
from src.llm import rerank_texts
from src.retrieval.graph_ops import get_one_hop_path


def _initialize_beam(entities, beam_width=15):
    """Initialize beam structure."""
    start_nodes = entities[:beam_width]
    return [
        {
            "nodes": [node],
            "edges": [],
            "paths": [],
            "weighted_score": node["weighted_score"],
            "is_used": False,
        }
        for node in start_nodes
    ]


def _extract_current_nodes_from_beam(beam):
    """Extract current terminal nodes from beam."""
    current_nodes = []
    unique_current_nodes = set()

    for b in beam:
        if not b["nodes"]:
            continue
        the_node_name = b["nodes"][-1]["entity_name"]
        the_path_score = b["weighted_score"]
        if the_node_name not in unique_current_nodes:
            current_nodes.append(
                {"entity_name": the_node_name, "weighted_score": the_path_score}
            )
            unique_current_nodes.add(the_node_name)

    return current_nodes


def _filter_visited_paths(hop_paths, visited_entities):
    """Filter out already-visited paths."""
    filtered_hop_paths = []
    for path in hop_paths:
        if path["tgt_name"] not in visited_entities:
            filtered_hop_paths.append(path)
            visited_entities.add(path["tgt_name"])
    return filtered_hop_paths


def _calculate_text_score(path, text_chunks):
    """Compute text score S_text(r, e) by summing max chunk scores for relation and entity."""
    rel_scores = [
        chunk["weighted_score"]
        for chunk in text_chunks[:30]
        if chunk["id"] in path["rel_chunk_id"]
    ]
    max_rel_score = max(rel_scores) if rel_scores else 0.0

    ent_scores = [
        chunk["weighted_score"]
        for chunk in text_chunks[:30]
        if chunk["id"] in path["ent_chunk_id"]
    ]
    max_ent_score = max(ent_scores) if ent_scores else 0.0

    return max_rel_score + max_ent_score


# ── Edge type weights for cross-modal beam search ──
EDGE_TYPE_WEIGHT = {
    "IDENTITY": 1.0,
    "DEPICTS": 0.7,
    "CONTAINS": 0.5,
    "VISUAL_RELATION": 0.6,
    "ADJACENT_TO_TEXT": 0.4,
}

# Keywords mapping for edge type detection
_EDGE_TYPE_KEYWORDS = {
    "identity equivalence": "IDENTITY",
    "likely depicts": "DEPICTS",
    "CONTAINS": "CONTAINS",
    "VISUAL_RELATION": "VISUAL_RELATION",
    "ADJACENT_TO_TEXT": "ADJACENT_TO_TEXT",
}


def _get_edge_type_weight(keywords: str, weights=None) -> float:
    """Extract edge type weight from relationship keywords.

    Image-derived edges store edge type in the keywords field
    (e.g., 'CONTAINS', 'VISUAL_RELATION', 'identity equivalence', 'likely depicts').
    Returns 1.0 (default) for edges without a recognized type.
    """
    if not keywords:
        return 1.0
    weights = weights or EDGE_TYPE_WEIGHT
    for kw, edge_type in _EDGE_TYPE_KEYWORDS.items():
        if kw.lower() in keywords.lower():
            return weights.get(edge_type, 1.0)
    return 1.0


def detect_edge_type(path) -> str:
    """Infer coarse multimodal edge type from existing KG edge metadata."""
    keywords = (path or {}).get("keywords", "") or ""
    for kw, edge_type in _EDGE_TYPE_KEYWORDS.items():
        if kw.lower() in keywords.lower():
            return edge_type
    src = (path or {}).get("src_name", "")
    tgt = (path or {}).get("tgt_name", "")
    if src.startswith(("IMAGE:", "VISUAL_ENTITY:")) and not tgt.startswith(
        ("IMAGE:", "VISUAL_ENTITY:")
    ):
        return "BRIDGE_V2T"
    if tgt.startswith(("IMAGE:", "VISUAL_ENTITY:")) and not src.startswith(
        ("IMAGE:", "VISUAL_ENTITY:")
    ):
        return "BRIDGE_T2V"
    return "TEXT_RELATION"


def _node_modality(node_name: str) -> str:
    if node_name.startswith(("IMAGE:", "VISUAL_ENTITY:")):
        return "visual"
    return "text"


def infer_path_action(path) -> str:
    """Map a KG edge to the BridgeRAG action it most likely performs."""
    edge_type = detect_edge_type(path)
    src_modality = _node_modality(path.get("src_name", ""))
    tgt_modality = _node_modality(path.get("tgt_name", ""))
    if edge_type in {"IDENTITY", "DEPICTS", "BRIDGE_V2T"}:
        return "BRIDGE_V2T" if src_modality == "visual" else "BRIDGE_T2V"
    if edge_type == "BRIDGE_T2V":
        return "BRIDGE_T2V"
    if edge_type in {"CONTAINS", "VISUAL_RELATION"} or (
        src_modality == "visual" and tgt_modality == "visual"
    ):
        return "VISUAL_REASON"
    return "TEXT_REASON"


def _plan_steps(plan):
    return [s for s in (plan or {}).get("steps", []) if s.get("action") != "ANSWER"]


def _action_matches_plan(path, step) -> bool:
    if not step:
        return False
    expected = step.get("action", "")
    actual = infer_path_action(path)
    edge_type = detect_edge_type(path)
    allowed_edges = set(step.get("allowed_edge_types") or [])
    if expected == actual:
        return True
    if expected in {"BRIDGE_V2T", "BRIDGE_T2V"} and edge_type in {
        "IDENTITY",
        "DEPICTS",
        "BRIDGE_V2T",
        "BRIDGE_T2V",
    }:
        return True
    if edge_type in allowed_edges:
        return True
    if expected == "VISUAL_GROUND" and edge_type == "CONTAINS":
        return True
    if expected == "TEXT_GROUND" and actual == "TEXT_REASON":
        return True
    return False


def _ground_score(path) -> float:
    source_ids = set(path.get("rel_chunk_id") or []) | set(path.get("ent_chunk_id") or [])
    if not source_ids:
        return 0.0
    return min(1.0, len(source_ids) / 2.0)


def _bridge_score(path) -> float:
    edge_type = detect_edge_type(path)
    if edge_type == "IDENTITY":
        return 1.0
    if edge_type == "DEPICTS":
        return 0.7
    if edge_type in {"BRIDGE_V2T", "BRIDGE_T2V"}:
        return 0.6
    return 0.0


def _clip_for_rerank(text, max_chars=900) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _path_to_rerank_text(path, max_chars=900) -> str:
    parts = [
        path.get("src_name", ""),
        path.get("keywords", ""),
        path.get("description", ""),
        path.get("tgt_name", ""),
        path.get("tgt_description", ""),
    ]
    return _clip_for_rerank(" | ".join(str(p or "") for p in parts), max_chars)


def _query_relation_terms(query: str):
    lowered = (query or "").lower()
    groups = {
        "origin": ["origin", "originate", "country", "from which country", "nationality"],
        "basionym": ["basionym", "originally described", "synonym"],
        "food": ["food", "source", "eat", "diet", "prey", "feed"],
        "inventor": ["inventor", "discoverer", "invent", "discover", "method"],
        "preserve": ["preserve", "canning", "stored", "storage"],
        "born": ["born", "birth", "year"],
    }
    terms = []
    for label, keywords in groups.items():
        if any(keyword in lowered for keyword in keywords):
            terms.extend(keywords)
            terms.append(label)
    for token in re.findall(r"[A-Za-z][A-Za-z-]{3,}", lowered):
        if token not in terms:
            terms.append(token)
    return terms[:30]


def _relation_hint_score(query: str, path, step=None) -> float:
    text = " ".join(
        str(path.get(key, "") or "")
        for key in ("keywords", "description", "tgt_name", "tgt_description")
    ).lower()
    hints = _query_relation_terms(query)
    relation_hint = ((step or {}).get("relation_hint") or "").lower()
    if relation_hint:
        hints.extend(re.findall(r"[a-z][a-z-]{3,}", relation_hint))
    if not hints or not text:
        return 0.0
    hits = sum(1 for term in set(hints) if term and term in text)
    return min(0.8, hits * 0.12)


def _initialize_plan_beam(entities, plan, beam_width=15):
    start_nodes = entities[:beam_width]
    steps = _plan_steps(plan)
    first_action = steps[0].get("action") if steps else ""
    starts_with_visual = first_action in {"VISUAL_GROUND", "VISUAL_REASON"}
    starts_with_text = first_action == "TEXT_GROUND"
    if starts_with_visual:
        anchor_nodes = [node for node in entities if node.get("query_visual_anchor")]
        named_anchor_nodes = [node for node in entities if node.get("query_anchor") and not node.get("query_visual_anchor")]
        visual_nodes = [
            node for node in entities if node.get("entity_name", "").startswith(("IMAGE:", "VISUAL_ENTITY:"))
        ]
        text_nodes = [
            node for node in entities if not node.get("entity_name", "").startswith(("IMAGE:", "VISUAL_ENTITY:"))
        ]
        start_nodes = (
            anchor_nodes + visual_nodes + named_anchor_nodes
        )[:beam_width] if (anchor_nodes or visual_nodes or named_anchor_nodes) else text_nodes[:beam_width]
    elif starts_with_text:
        text_nodes = [
            node for node in entities if not node.get("entity_name", "").startswith(("IMAGE:", "VISUAL_ENTITY:"))
        ]
        start_nodes = text_nodes[:beam_width] if text_nodes else entities[:beam_width]

    def _initial_plan_state(node):
        node_name = node.get("entity_name", "")
        node_is_visual_entity = node_name.startswith("VISUAL_ENTITY:")
        node_is_text = not node_name.startswith(("IMAGE:", "VISUAL_ENTITY:"))
        if first_action == "VISUAL_GROUND" and node.get("query_visual_anchor"):
            return 2, ["VISUAL_GROUND", "BRIDGE_V2T"], 2.0
        if first_action == "VISUAL_GROUND" and node_is_visual_entity:
            return 1, ["VISUAL_GROUND"], 1.0
        if first_action == "TEXT_GROUND" and node_is_text:
            return 1, ["TEXT_GROUND"], 1.0
        return 0, [], 0.0

    initial_items = []
    for node in start_nodes:
        plan_step, matched_actions, consistency = _initial_plan_state(node)
        weighted_score = node.get("weighted_score", 0.0) + consistency
        initial_items.append(
            {
                "nodes": [node],
                "edges": [],
                "paths": [],
                "plan_step": plan_step,
                "matched_actions": matched_actions,
                "weighted_score": weighted_score,
                "score_breakdown": {
                    "retrieval": node.get("weighted_score", 0.0),
                    "path": 0.0,
                    "bridge": 0.0,
                    "ground": 0.0,
                    "consistency": consistency,
                    "cost": 0.0,
                },
                "is_used": False,
            }
        )
    return initial_items


def _extend_plan_beam_item(beam_item, path, score_breakdown, plan_step, matched_action):
    node = {
        "entity_name": path["tgt_name"],
        "description": path.get("tgt_description", ""),
    }
    edge = {
        "src_name": path["src_name"],
        "description": path.get("description", ""),
        "tgt_name": path["tgt_name"],
        "keywords": path.get("keywords", ""),
        "edge_type": detect_edge_type(path),
        "source_ids": list(set(path.get("rel_chunk_id") or [])),
    }
    breakdown = beam_item["score_breakdown"].copy()
    for key, value in score_breakdown.items():
        breakdown[key] = breakdown.get(key, 0.0) + value
    matched_actions = beam_item["matched_actions"].copy()
    if matched_action:
        matched_actions.append(matched_action)
    return {
        "nodes": beam_item["nodes"] + [node],
        "edges": beam_item["edges"] + [edge],
        "paths": beam_item["paths"] + [path],
        "plan_step": plan_step,
        "matched_actions": matched_actions,
        "weighted_score": sum(breakdown.values()),
        "score_breakdown": breakdown,
        "is_used": False,
    }


async def modality_constrained_beam_search(
    query,
    entities,
    text_chunks,
    knowledge_graph_inst,
    plan,
    max_hops=3,
    beam_width=15,
    alpha=1.0,
    beta=1.0,
    decay=0.9,
    rerank_texts_func=rerank_texts,
    edge_type_weights=None,
    soft_penalty=0.35,
    bridge_bonus=1.0,
    ground_bonus=0.5,
    rerank_max_paths=150,
    rerank_max_chars=900,
):
    """Bridge-aware soft modality-constrained beam search.

    The plan is used as a soft guide: matching actions advance the plan state
    and receive a consistency bonus; non-matching hops are retained with a
    penalty so predicted-plan errors do not prune useful paths outright.
    """
    if not entities:
        return []

    plan_steps = _plan_steps(plan)
    beam = _initialize_plan_beam(entities, plan, beam_width)

    for hop in range(max_hops):
        current_nodes = _extract_current_nodes_from_beam(beam)
        if not current_nodes:
            break
        hop_paths = await get_one_hop_path(
            [node["entity_name"] for node in current_nodes], knowledge_graph_inst
        )
        if not hop_paths:
            break

        terminal_names = {item["nodes"][-1]["entity_name"] for item in beam if item.get("nodes")}
        candidate_paths = [
            path for path in hop_paths
            if path.get("src_name") in terminal_names and path.get("tgt_name") not in terminal_names
        ]
        if not candidate_paths:
            break

        sample_size = min(rerank_max_paths, len(candidate_paths))
        random.seed(52 + hop)
        sampled_paths = random.sample(candidate_paths, sample_size)
        hop_path_strs = [
            _path_to_rerank_text(p, rerank_max_chars)
            for p in sampled_paths
        ]
        hop_ranks = await rerank_texts_func(query=query, texts=hop_path_strs)
        normalize_list(hop_ranks, key="relevance_score")

        for path in sampled_paths:
            path["text_score"] = _calculate_text_score(path, text_chunks)
        normalize_list(sampled_paths, key="text_score")

        candidates = []
        weights = {**EDGE_TYPE_WEIGHT, **(edge_type_weights or {})}
        for path, rank in zip(sampled_paths, hop_ranks):
            for beam_item in beam:
                if not beam_item["nodes"]:
                    continue
                if beam_item["nodes"][-1]["entity_name"] != path["src_name"]:
                    continue
                if any(n["entity_name"] == path["tgt_name"] for n in beam_item["nodes"]):
                    continue

                step_idx = min(beam_item.get("plan_step", 0), len(plan_steps) - 1)
                current_step = plan_steps[step_idx] if plan_steps else {}
                matches = _action_matches_plan(path, current_step)
                next_step = beam_item.get("plan_step", 0) + 1 if matches else beam_item.get("plan_step", 0)
                consistency = 1.0 if matches else -abs(soft_penalty)
                bridge = _bridge_score(path) * bridge_bonus
                ground = _ground_score(path) * ground_bonus
                edge_wt = _get_edge_type_weight(path.get("keywords", ""), weights)
                path_score = (
                    alpha * rank.get("relevance_score", 0.0)
                    + beta * path.get("text_score", 0.0)
                    + _relation_hint_score(query, path, current_step)
                ) * (decay**hop) * edge_wt
                score_breakdown = {
                    "path": path_score,
                    "bridge": bridge,
                    "ground": ground,
                    "consistency": consistency,
                    "cost": -0.05,
                }
                matched_action = current_step.get("action", "") if matches else ""
                candidates.append(
                    _extend_plan_beam_item(
                        beam_item, path, score_breakdown, next_step, matched_action
                    )
                )

        if not candidates:
            break
        candidates.sort(key=lambda x: x["weighted_score"], reverse=True)
        beam = candidates[:beam_width]

    return sorted(beam, key=lambda x: x["weighted_score"], reverse=True)


async def _calculate_path_scores(
    query,
    filtered_hop_paths,
    text_chunks,
    current_nodes,
    hop=0,
    alpha=1.0,
    beta=1.0,
    decay=0.9,
    rerank_texts_func=rerank_texts,
    edge_type_weights=None,  # dict config override
):
    """
    Calculate path scores combining structural and text scores with decay.

    S_hop = alpha * S_struct + beta * S_text
    score(p_h) = score(p_{h-1}) + decay^(h-1) * S_hop * edge_type_weight
    """
    if not filtered_hop_paths:
        return []

    sample_size = min(500, len(filtered_hop_paths))

    random.seed(52)
    sampled_paths = random.sample(filtered_hop_paths, sample_size)
    hop_path_strs = [
        p["src_name"] + p["description"] + p["tgt_description"] for p in sampled_paths
    ]

    hop_ranks = await rerank_texts_func(query=query, texts=hop_path_strs)

    normalize_list(hop_ranks, key="relevance_score")

    for path in sampled_paths:
        path["text_score"] = _calculate_text_score(path, text_chunks)
    normalize_list(sampled_paths, key="text_score")

    decay_factor = decay**hop

    # Merge config weights with defaults
    weights = {**EDGE_TYPE_WEIGHT, **(edge_type_weights or {})}

    for path, rank in zip(sampled_paths, hop_ranks):
        s_struct = rank["relevance_score"]
        s_text = path["text_score"]
        s_hop = alpha * s_struct + beta * s_text

        # Apply edge type weighting
        edge_wt = _get_edge_type_weight(path.get("keywords", ""), weights)
        s_hop *= edge_wt

        prev_score = 0.0
        for node in current_nodes:
            if node["entity_name"] == path["src_name"]:
                prev_score = node["weighted_score"]
                break

        path["weighted_score"] = prev_score + decay_factor * s_hop

    normalize_list(sampled_paths, key="weighted_score")

    return sorted(sampled_paths, key=lambda x: x["weighted_score"], reverse=True)


def _create_new_beam_item(beam_item, path):
    """Create a new beam item by extending with a path."""
    new_nodes = beam_item["nodes"].copy()
    new_nodes.append(
        {
            "entity_name": path["tgt_name"],
            "description": path["tgt_description"],
        }
    )

    new_edges = beam_item["edges"].copy()
    new_edges.append(
        {
            "src_name": path["src_name"],
            "description": path["description"],
            "tgt_name": path["tgt_name"],
            "keywords": path["keywords"],
        }
    )

    new_paths = beam_item["paths"].copy()
    new_paths.append(path)

    return {
        "nodes": new_nodes,
        "edges": new_edges,
        "paths": new_paths,
        "weighted_score": path["weighted_score"],
        "is_used": False,
    }


def _update_beam(beam, filtered_hop_paths, beam_width=15):
    """Update beam with new scored paths."""
    new_beam = []

    top_paths = filtered_hop_paths[:beam_width]

    for path in top_paths:
        src_entity = path["src_name"]
        for b in beam:
            if b["nodes"] and b["nodes"][-1]["entity_name"] == src_entity:
                new_beam.append(_create_new_beam_item(b, path))
                b["is_used"] = True

    if len(new_beam) < beam_width:
        remaining = beam_width - len(new_beam)
        unused = [b for b in beam if not b["is_used"]]
        unused.sort(key=lambda x: x["weighted_score"], reverse=True)
        new_beam.extend(unused[:remaining])

    return new_beam[:beam_width]


async def beam_search(
    query,
    entities,
    text_chunks,
    knowledge_graph_inst,
    max_hops=3,
    beam_width=15,
    alpha=1.0,
    beta=1.0,
    decay=0.9,
    rerank_texts_func=rerank_texts,
    edge_type_weights=None,  # dict: edge type → weight multiplier
):
    """
    Multi-hop beam search on the knowledge graph.

    Iteratively expands terminal nodes, scores paths using structural
    and text relevance with decay, and retains top-W paths.
    Supports edge type weighting for cross-modal hops.
    """
    beam = _initialize_beam(entities, beam_width)
    visited_entities = {node["entity_name"] for node in entities[:beam_width]}

    for hop in range(max_hops):
        current_nodes = _extract_current_nodes_from_beam(beam)
        if not current_nodes:
            break

        hop_paths = await get_one_hop_path(
            [node["entity_name"] for node in current_nodes], knowledge_graph_inst
        )

        filtered_hop_paths = _filter_visited_paths(hop_paths, visited_entities)
        if not filtered_hop_paths:
            break

        scored_paths = await _calculate_path_scores(
            query,
            filtered_hop_paths,
            text_chunks,
            current_nodes,
            hop=hop,
            alpha=alpha,
            beta=beta,
            decay=decay,
            rerank_texts_func=rerank_texts_func,
            edge_type_weights=edge_type_weights,
        )

        beam = _update_beam(beam, scored_paths, beam_width)

    return beam


def _extract_from_complete_paths(beam, final_entities, final_relationships, max_hops):
    """Extract entities and relations from complete beam paths."""
    if beam is None:
        return

    for b in beam:
        if len(b["nodes"]) >= 1:
            for node in b["nodes"]:
                if node["entity_name"] not in [
                    e["entity_name"] for e in final_entities
                ]:
                    final_entities.append(node)

            for edge in b["edges"]:
                if edge["description"] not in [
                    e["description"] for e in final_relationships
                ]:
                    final_relationships.append(edge)


def _format_context(entities, relationships, chunks):
    """Format context string from entities, relationships, and chunks."""
    str_entities = "\n".join(
        [f"{e['entity_name']}: {e['description']}" for e in entities]
    )
    str_relationships = "\n".join(
        [f"{r['src_name']} {r['description']} {r['tgt_name']}" for r in relationships]
    )
    str_chunks = "\n".join([f"{c['content']}" for c in chunks])

    return str_chunks + "\n\n" + str_relationships + "\n\n" + str_entities


async def process_beam_output(
    beam,
    entities,
    relationships,
    text_chunks,
    max_hops,
    max_entities_tokens=4000,
    max_relationships_tokens=4000,
    max_chunks_tokens=4000,
    # ── Image modality (optional) ──
    image_chunks_db=None,
    image_captions=None,
):
    """Process beam search output into truncated context.

    Optionally includes image captions from entities' source_id chains.
    """
    entities = entities or []
    relationships = relationships or []
    text_chunks = text_chunks or []

    final_entities = entities[:15]
    final_relationships = relationships[:15]

    _extract_from_complete_paths(beam, final_entities, final_relationships, max_hops)

    # ── Extract image captions from entity source_ids ──
    image_contexts = []
    if image_chunks_db is not None:
        all_source_ids = set()
        for e in final_entities:
            # Entities may have source_id from their node data
            pass
        # Use provided image_captions if available
        if image_captions:
            for img_id, caption in image_captions:
                image_contexts.append(f"[Image Caption: {img_id}]: {caption}")
    elif image_captions:
        for img_id, caption in image_captions:
            image_contexts.append(f"[Image Caption: {img_id}]: {caption}")

    final_entities = await truncate_list_by_token_size_async(
        final_entities, lambda x: x["description"], max_token_size=max_entities_tokens
    )
    final_relationships = await truncate_list_by_token_size_async(
        final_relationships,
        lambda x: x["description"],
        max_token_size=max_relationships_tokens,
    )
    final_chunks = await truncate_by_token_size_async(
        text_chunks, lambda x: x["content"], max_token_size=max_chunks_tokens
    )

    context = _format_context(final_entities, final_relationships, final_chunks)

    # Append image context
    if image_contexts:
        context += "\n\n[Image Descriptions]\n" + "\n".join(image_contexts)

    return final_entities, final_relationships, final_chunks, context
