"""Query execution entry points."""

import ast
import json
import re
from functools import lru_cache
from pathlib import Path

from src.utils import parse_json_response
from src.pipeline.context_packing import (encode as encode_string_by_tiktoken, decode as decode_tokens_by_tiktoken, pack_text, input_budget)
from src.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage, TextChunkSchema
from src.prompt import PROMPTS
from src.pipeline.mm_sgn import _run_image_retrieve, _run_text_sgn_once, _page_title_from_caption
from src.pipeline.simple_state import SubQuestionState


def _simple_answer_text(answer: object) -> str:
    """Preserve valid false/zero values instead of applying truthiness defaults."""
    if answer is None:
        return ""
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    if isinstance(answer, (list, dict)):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer).strip()


def _is_unknown_answer(answer: object) -> bool:
    if isinstance(answer, (list, tuple, dict, set)) and not answer:
        return True
    text = _simple_answer_text(answer).lower()
    return not text or text in {"i don't know", "i don't know.", "unknown", "n/a", "none", "null"} or bool(re.fullmatch(r"\[\s*\]|\{\s*\}", text))


SIMPLE_RETRIEVAL_STRATEGIES = {
    "text_only",
    "image_only",
    "text_then_image",
    "image_then_text",
    "both_parallel",
    "direct_answer_if_enough",
}
SIMPLE_ABLATION_VARIANTS = {"full", "sgn_only", "without_sgn", "without_ac"}


@lru_cache(maxsize=2)
def _simple_load_plan_replay(path: str) -> dict:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Plan replay file must contain a list of saved Full results")
    plans = {}
    for row in rows:
        question = str(row.get("query") or "")
        plan = (row.get("answer") or {}).get("simple_plan")
        if not question or not isinstance(plan, dict) or not plan.get("sub_questions"):
            raise ValueError("Plan replay file has a missing question or decomposition")
        if question in plans:
            raise ValueError("Plan replay file contains duplicate questions")
        plans[question] = plan
    return plans


async def _simple_plain_text_retrieve(question: str, stores: dict, funcs: dict, config) -> dict:
    """Dense chunk retrieval with no graph/entity/relationship access."""
    rows = await stores["chunks_vdb"].query(question, top_k=int(config.kg_top_k))
    chunks = []
    for row in rows:
        chunk = await stores["text_chunks_db"].get_by_id(row.get("id") or row.get("__id__"))
        if chunk and chunk.get("content"):
            chunks.append({"content": str(chunk["content"])})
    context = pack_text(question, chunks, int(config.mm_sgn_context_max_tokens))
    prompt = PROMPTS["answer_sub_query"].format(sub_query=question, context=context)
    try:
        response = await funcs["llm_model_func"](prompt=prompt, temperature=0.0, max_tokens=512)
        parsed = await parse_json_response(response, funcs["llm_model_func"])
    except Exception:
        parsed = {}
    parsed = parsed if isinstance(parsed, dict) else {}
    return {"answer": _simple_answer_text(parsed.get("answer")) or "I don't know.",
            "evidence": [{"type": "context", "content": context}] if context else [],
            "candidate_state": {"candidates": []}, "retrieval_backend": "plain_dense_text"}


async def _simple_plain_image_retrieve(question: str, stores: dict, config) -> dict:
    """Dense image retrieval with stored metadata, without KG enrichment."""
    rows = await stores["images_vdb"].query(question, top_k=int(config.image_retrieval_candidate_k))
    candidates = []
    for row in rows:
        image_id = row.get("id") or row.get("__id__")
        chunk = await stores["image_chunks_db"].get_by_id(image_id) if image_id else None
        chunk = chunk or {}
        caption = str(chunk.get("caption") or row.get("caption") or "")
        candidates.append({"image_id": image_id or "", "image_path": chunk.get("image_path") or row.get("image_path") or "",
                           "page_title": _page_title_from_caption(caption), "caption": caption,
                           "global_description": chunk.get("global_description") or "", "linked_entities": [],
                           "fusion_score": float(row.get("__metrics__", row.get("distance", 0.0)) or 0.0)})
    return {"answer": "I don't know.", "candidate_state": {"candidates": candidates},
            "evidence": [], "retrieval_backend": "plain_dense_image"}


def _safe_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _normalize_simple_plan(parsed: object, query: str, max_nodes: int = 6) -> dict:
    parsed = parsed if isinstance(parsed, dict) else {}
    raw_nodes = (
        parsed.get("sub_queries")
        or parsed.get("sub_questions")
        or parsed.get("Q_L")
        or parsed.get("nodes")
        or []
    )
    nodes = []
    for idx, raw in enumerate(raw_nodes[:max_nodes], start=1):
        if not isinstance(raw, dict):
            raw = {"question": str(raw)}
        question = str(raw.get("question") or raw.get("query") or raw.get("natural_language_subquestion") or "").strip()
        if not question:
            continue
        deps = []
        for dep in raw.get("depends_on") or []:
            try:
                dep_id = int(dep)
            except Exception:
                continue
            if 0 < dep_id < idx:
                deps.append(dep_id)
        # A comparison can reference entity answers as well as numeric operands.
        # The planner's explicit dependency list sometimes contains only operands.
        deps.extend(int(ref) for ref in re.findall(r"\{answer_(\d+)\}", question) if 0 < int(ref) < idx)
        deps = sorted(set(deps))
        nodes.append(
            {
                "id": len(nodes) + 1,
                "question": question,
                "has_dependency": bool(deps),
                "depends_on": deps,
            }
        )
    if not nodes:
        nodes = [{"id": 1, "question": query, "depends_on": []}]
    id_map = {node["id"]: idx + 1 for idx, node in enumerate(nodes)}
    normalized = []
    for idx, node in enumerate(nodes, start=1):
        normalized.append(
            {
                "id": idx,
                "question": str(node["question"]),
                "has_dependency": bool(node.get("depends_on")),
                "depends_on": [id_map.get(dep, dep) for dep in node.get("depends_on", []) if id_map.get(dep, dep) < idx],
            }
        )
    return {
        "reasoning_plan": str(parsed.get("reasoning_plan") or "Natural-language logical decomposition."),
        "sub_questions": normalized,
    }


def _simple_ready_nodes(plan: dict, executed: set[int]) -> list[dict]:
    ready = []
    for node in plan.get("sub_questions") or []:
        node_id = int(node.get("id") or 0)
        if node_id in executed:
            continue
        if all(int(dep) in executed for dep in node.get("depends_on") or []):
            ready.append(node)
    return ready


def _compact_text_evidence(result: dict, limit: int = 8) -> list[dict]:
    evidence = []
    for item in (result or {}).get("evidence") or []:
        evidence.append(
            {
                "type": item.get("type", "context"),
                "content": str(item.get("content") or item),
            }
        )
    return evidence[:limit]


def _compact_image_evidence(result: dict, limit: int = 4) -> list[dict]:
    output = []
    state = (result or {}).get("candidate_state") or {}
    candidates = state.get("candidates") or []
    if candidates:
        for candidate in candidates[:limit]:
            output.append(
                {
                    "image_id": candidate.get("image_id", ""),
                    "image_path": candidate.get("image_path", ""),
                    "page_title": candidate.get("page_title", ""),
                    "linked_entities": candidate.get("linked_entities") or [],
                    "aliases": candidate.get("aliases") or [],
                    "caption": candidate.get("caption", ""),
                    "global_description": candidate.get("global_description", ""),
                    "fusion_score": candidate.get("fusion_score", 0.0),
                }
            )
    if not output:
        for item in (result or {}).get("evidence") or []:
            if str(item.get("type") or "").startswith("kb_image") or item.get("image_path"):
                kg_context = item.get("kg_context") or {}
                output.append(
                    {
                        "image_id": item.get("id") or item.get("__id__") or "",
                        "image_path": item.get("image_path") or kg_context.get("image_path", ""),
                        "page_title": item.get("page_title") or "",
                        "linked_entities": kg_context.get("linked_entities") or [],
                        "aliases": item.get("aliases") or [],
                        "caption": item.get("caption") or kg_context.get("caption", ""),
                        "global_description": item.get("global_description", ""),
                        "fusion_score": item.get("fusion_score", item.get("__metrics__", 0.0)),
                    }
                )
                if len(output) >= limit:
                    break
    return output[:limit]


def _format_simple_evidence_for_prompt(text_evidence: list[dict], image_evidence: list[dict], visual_observations: list[dict], *, query: str = "", max_tokens: int = 4000) -> str:
    visual = "## VLM visual observations\n" + json.dumps(_compact_visual_observations(visual_observations), ensure_ascii=False)
    # Compact records contain every inspected candidate, not only the first four.
    visual_budget = min(len(encode_string_by_tiktoken(visual)), max_tokens // 2)
    if len(encode_string_by_tiktoken(visual)) > visual_budget:
        visual = "## VLM visual observations\n" + json.dumps([
            {"candidate": v.get("candidate_set_value") or v.get("candidate_entity") or v.get("page_title", ""),
             "result": v.get("result", ""),
             "observation": _simple_token_truncate(v.get("observation") or v.get("visual_observation") or "", 24)}
            for v in visual_observations], ensure_ascii=False)
    if len(encode_string_by_tiktoken(visual)) > max_tokens // 2:
        visual = "## VLM visual observations\n" + json.dumps([
            {"candidate": v.get("candidate_set_value") or v.get("candidate_entity") or v.get("page_title", ""),
             "result": v.get("result", "")} for v in visual_observations], ensure_ascii=False)
    if len(encode_string_by_tiktoken(visual)) > max_tokens - 100:
        raise ValueError("Candidate observations exceed evidence budget; refusing to silently drop candidates")
    text_budget = max(0, max_tokens - len(encode_string_by_tiktoken(visual)) - 100)
    text = pack_text(query, text_evidence, text_budget)
    result = visual + "\n\n## Text/KG evidence\n" + text
    # Titles/captions are supporting metadata, subordinate to inspected observations.
    remaining = max_tokens - len(encode_string_by_tiktoken(result)) - 16
    if remaining > 0 and image_evidence:
        metadata = json.dumps([{"title": v.get("page_title", ""), "image_id": v.get("image_id", ""),
                                "description": v.get("global_description", "")} for v in image_evidence], ensure_ascii=False)
        result += "\n## Image metadata\n" + _simple_token_truncate(metadata, remaining)
    return _simple_token_truncate(result, max_tokens)


def _compact_visual_observations(observations: list[dict]) -> list[dict]:
    records = []
    for item in observations:
        if item.get("type") == "visual_selector_disambiguation":
            records.append({"selected_candidate": item.get("selected_candidate", ""), "reason": str(item.get("rationale", ""))[:160]})
            continue
        records.append({
            "candidate": item.get("candidate_set_value") or item.get("candidate_entity") or item.get("page_title", ""),
            "image_id": item.get("image_id", ""),
            "result": item.get("result", ""),
            "observation": str(item.get("observation") or item.get("visual_observation") or "")[:200],
        })
    return records


def _simple_question_expects_candidate_set(question: str) -> bool:
    text = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    markers = [
        "what film titles",
        "what films",
        "which films",
        "what tv shows",
        "what television shows",
        "which tv shows",
        "which television shows",
        "what titles",
        "which titles",
        "what clubs",
        "which clubs",
        "what teams",
        "which teams",
        "what players",
        "which players",
        "what authors",
        "which authors",
        "what are the common names",
        "who were",
        "which of",
        "from the list",
        "candidate",
    ]
    return any(marker in text for marker in markers)


def _split_candidate_answer(answer: str, question: str, limit: int = 12) -> list[str]:
    text = _simple_answer_text(answer)
    if not text or _is_unknown_answer(text):
        return []
    if not _simple_question_expects_candidate_set(question):
        return [text]
    if text.startswith("["):
        try:
            values = json.loads(text)
        except (ValueError, TypeError):
            try:
                values = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                values = None
        if isinstance(values, (list, tuple)) and all(isinstance(v, str) for v in values):
            return _dedupe_preserve_order([v.strip() for v in values if v.strip()])[:limit]
    cleaned = re.sub(r"^(?:the candidates are|the films are|the titles are|the teams are|the players are|the authors are)[:\s]+", "", text, flags=re.I)
    # Reader answers can have a prose prefix before the actual list.
    if ":" in cleaned and re.match(r"^.+?\b(?:shows|films|titles|authors|teams|candidates)\s*:", cleaned, re.I):
        cleaned = cleaned.split(":", 1)[1].strip()
    parts = re.split(r"\s*[,;]\s*|\s+\|\s+", cleaned)
    output = []
    for part in parts:
        item = re.sub(r"^\s*(?:and|or)\s+", "", part.strip(), flags=re.I)
        item = re.sub(r"\s+\([^)]*$", "", item).strip().rstrip(".")
        if not item or len(item.split()) > 12:
            continue
        if item.lower() in {"and", "or", "none"}:
            continue
        output.append(item)
        if len(output) >= limit:
            break
    return _dedupe_preserve_order(output) or [text]


def _simple_extract_candidate_set_from_question(question: str, limit: int = 12) -> list[str]:
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    # Parse the structured list itself; words such as 'has' can occur in titles.
    if re.search(r"\b(?:among|which of|from)\b", text, re.I):
        start = text.find("[")
        if start >= 0:
            try:
                values, _ = json.JSONDecoder().raw_decode(text[start:])
            except ValueError:
                values = None
            if isinstance(values, list) and all(isinstance(v, str) for v in values):
                return _dedupe_preserve_order(values)[:limit]
    candidates_text = ""
    patterns = [
        r"which of (.+?) (?:has|have|features|feature|shows|show|contains|contain|wears|wear|with) ",
        r"which (?:bird|title|film|movie|television show|tv title|team|school) from (.+?) (?:has|have|features|feature|shows|show|contains|contain|wears|wear|with) ",
        r"from (.+?),? which ",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            candidates_text = match.group(1).strip(" .,:;")
            break
    if not candidates_text:
        return []
    return _split_candidate_answer(candidates_text, "which of candidate list", limit=limit)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        key = re.sub(r"\W+", " ", str(value).casefold()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(str(value).strip())
    return output


def _simple_visual_predicate(question: str) -> str:
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    patterns = [
        r"which\s+(?:of|\w+\s+from)\s+.+?\s+(wears?\s+[^?]+)",
        r"which\s+(?:of|(?:bird|title|film|movie|team|school)\s+from)\s+.+?\s+(?:has|have|featuring|features|showing|shows|contains|contain|with)\s+([^?]+)",
        r"does\s+(?:the\s+)?(?:image|photo|picture|poster|cover|logo)\s+(?:show|have|feature|contain|include)\s+([^?]+)",
        r"(?:poster|cover|logo|image|photo|picture|title screen)\s+(?:that|which)?\s*(?:has|have|featuring|features|showing|shows|contains|contain|with)\s+([^?]+)",
        r"(?:has|have|featuring|features|showing|shows|contains|contain|with)\s+([^?]+?)\s+(?:on|in)\s+(?:the\s+)?(?:poster|cover|logo|image|photo|picture|title screen)",
        r"that\s+(?:has|have|features|shows|contains)\s+([^?]+)",
        r"with\s+([^?]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            pred = match.group(1).strip(" .,:;")
            if pred:
                return pred
    return text


def _simple_has_visual_predicate(question: str) -> bool:
    text = str(question or "").lower()
    visual_phrase_patterns = [
        r"\b(?:poster|cover|logo|image|photo|picture|title screen)\s+(?:that\s+|which\s+)?(?:shows?|features?|contains?|includes?|with)\b",
        r"\b(?:shows?|features?|contains?|includes?)\s+(?:a|an|the|one|two|at least|white|black|red|green|blue|yellow|gray|grey)\b",
    ]
    if any(re.search(pattern, text) for pattern in visual_phrase_patterns):
        return True
    visual_markers = [
        "poster",
        "cover",
        "logo",
        "image",
        "photo",
        "picture",
        "wearing",
        "holding",
        "stripe",
        "color",
        "colour",
        "beard",
        "eyeglasses",
        "glasses",
        "standing next to",
        "near his mouth",
        "near her mouth",
        "featuring",
        "showing",
    ]
    return any(marker in text for marker in visual_markers)


def _simple_question_is_visual_selector(question: str) -> bool:
    text = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    if "which of " in text or re.search(r"\bfrom .+?,?\s+which\b", text):
        return True
    if not _simple_has_visual_predicate(text):
        return False
    if re.search(r"\b(?:who|which)\b.*\bamong\b", text):
        return True
    selector_markers = [
        "which bird",
        "which title",
        "which film",
        "which movie",
        "which television",
        "which tv",
        "which team",
        "which school",
        "which logo",
        "what title",
        "what film",
        "what movie",
        "what television show",
        "what tv title",
        "what video game title",
    ]
    return any(marker in text for marker in selector_markers)


def _simple_norm_name(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(value or "").casefold())).strip()


def _simple_image_matches_candidate_name(name: str, item: dict) -> bool:
    """Check that a candidate-specific image search actually returned that candidate.

    Candidate-expanded visual search asks queries such as ``Ali train poster``.
    CLIP can return a visually perfect train image whose page is unrelated to
    Ali; without this guard the visual verifier turns that unrelated image into
    false evidence for Ali.
    """

    name_key = _simple_norm_name(name)
    if not name_key:
        return False
    surfaces = [
        str(item.get("page_title") or ""),
        str(item.get("candidate_entity") or ""),
    ]
    surfaces.extend(str(entity) for entity in (item.get("linked_entities") or []) if str(entity).strip())
    surfaces.extend(str(alias) for alias in (item.get("aliases") or []) if str(alias).strip())
    for surface in surfaces:
        surface_key = _simple_norm_name(surface)
        if not surface_key:
            continue
        base_name = _simple_norm_name(re.sub(r"\s*\([^)]*\)\s*$", "", name))
        base_surface = _simple_norm_name(re.sub(r"\s*\([^)]*\)\s*$", "", surface))
        if name_key == surface_key or base_name == base_surface:
            return True
    return False


async def _simple_decide_strategy(question: str, original_query: str, previous_summary: str, funcs: dict) -> dict:
    prompt = PROMPTS["simple_retrieval_decision_prompt"].format(
        original_query=original_query,
        sub_question=question,
        previous_answers=previous_summary or "None",
    )
    try:
        response = await funcs["llm_model_func"](prompt=prompt, temperature=0.0, max_tokens=128)
        parsed = await parse_json_response(response, funcs["llm_model_func"])
    except Exception:
        parsed = {}
    parsed = parsed if isinstance(parsed, dict) else {}
    strategy = str(parsed.get("strategy") or "both_parallel").strip().lower()
    if strategy not in SIMPLE_RETRIEVAL_STRATEGIES:
        strategy = "both_parallel"
    # Visible properties cannot be established from a movie/entity name.
    if strategy in {"text_only", "direct_answer_if_enough"} and re.search(
        r"\b(?:what|which)\b.*\b(?:colou?r|holding|wearing|wears?|visible|eyeglasses)\b", question, re.I
    ):
        strategy = "image_only"
        parsed["image_query"] = question
        parsed["rationale"] = "The current question asks for visible image content."
    return {
        "strategy": strategy,
        "rationale": str(parsed.get("rationale") or ""),
        "text_query": str(parsed.get("text_query") or question).strip() or question,
        "image_query": str(parsed.get("image_query") or question).strip() or question,
    }


def _simple_register_image_aliases(result: dict, execution_context: dict) -> None:
    """Use explicit display-title links and qualified locations in retrieved text.

    No gold answers, global alias guesses, or substring identity matching.
    Table links are rendered as 'display (Canonical (disambiguation))'.
    """
    aliases = execution_context.setdefault("image_aliases", {})
    def add(canonical, alias):
        key = _simple_norm_name(canonical)
        if key and alias:
            aliases[key] = _dedupe_preserve_order(aliases.get(key, []) + [alias])
    for evidence in (result or {}).get("evidence", []):
        text = str(evidence.get("content", ""))
        for cell in re.split(r"[|\n]", text):
            cell = cell.strip()
            opening = cell.find(" (")
            if opening < 1 or not cell.endswith(")"):
                continue
            display, canonical = cell[:opening], cell[opening + 2:-1]
            # Require the nested title format, rather than treating ordinary
            # descriptive parentheses such as '(uncredited)' as entity links.
            if "(" in canonical and canonical.count("(") == canonical.count(")"):
                add(canonical, display)
                add(canonical, cell)
        for match in re.finditer(
            r"\bin ([A-Z][\w-]*(?: [A-Z][\w-]*){0,3}), ([A-Z][\w-]*(?: [A-Z][\w-]*){0,3})\b", text):
            add(match.group(1), match.group(1) + ", " + match.group(2))


async def _simple_text_retrieve(question: str, node_id: int, stores: dict, funcs: dict, config, execution_context: dict) -> dict:
    if getattr(config, "cor_simple_ablation_variant", "full") == "without_sgn":
        result = await _simple_plain_text_retrieve(question, stores, funcs, config)
        _simple_register_image_aliases(result, execution_context)
        return result
    node = {
        "id": node_id,
        "query": question,
        "natural_language_subquestion": question,
        "operation_type": "TEXT_RETRIEVE",
        "modality": "text",
        "input_anchor": question,
        "input_source": "text_kb",
        "expected_output_type": "answer",
        "allowed_edge_types": ["TEXT_RELATION"],
        "confidence": 0.7,
    }
    result = await _run_text_sgn_once(node, stores, funcs, config, execution_context)
    _simple_register_image_aliases(result, execution_context)
    return result


async def _simple_image_retrieve(question: str, node_id: int, stores: dict, funcs: dict, config, execution_context: dict) -> dict:
    if getattr(config, "cor_simple_ablation_variant", "full") == "without_sgn":
        return await _simple_plain_image_retrieve(question, stores, config)
    node = {
        "id": node_id,
        "query": question,
        "natural_language_subquestion": question,
        "operation_type": "IMAGE_RETRIEVE",
        "modality": "image",
        "input_anchor": question,
        "input_source": "kb_image",
        "expected_output_type": "entity",
        "allowed_edge_types": ["SOURCE", "DEPICTS", "CONTAINS"],
        "confidence": 0.7,
    }
    result = await _run_image_retrieve(node, stores, funcs, execution_context, config)
    aliases = execution_context.get("image_aliases", {})
    for item in ((result or {}).get("candidate_state") or {}).get("candidates", []):
        item["aliases"] = _dedupe_preserve_order((item.get("aliases") or []) + aliases.get(
            _simple_norm_name(item.get("page_title", "")), []))
    return result


async def _simple_visual_observe(sub_question: str, image_evidence: list[dict], funcs: dict) -> list[dict]:
    vlm_func = funcs.get("vlm_model_func")
    if not vlm_func or not image_evidence:
        return []
    observations = []
    prompt = PROMPTS["simple_visual_observation_prompt"].format(sub_question=sub_question)
    for item in image_evidence[:4]:
        image_path = item.get("image_path", "")
        if not image_path:
            continue
        try:
            response = await vlm_func(image_path=image_path, prompt=prompt, temperature=0.0, max_tokens=256)
            parsed = await parse_json_response(response, funcs["llm_model_func"])
        except Exception:
            parsed = {}
        parsed = parsed if isinstance(parsed, dict) else {}
        observation = str(parsed.get("visual_observation") or parsed.get("answer") or "").strip()
        observations.append(
            {
                "image_id": item.get("image_id", ""),
                "image_path": image_path,
                "page_title": item.get("page_title", ""),
                "candidate_entity": ", ".join([str(x) for x in (item.get("linked_entities") or [])[:3]]),
                "visual_observation": observation or "I don't know.",
                "rationale": str(parsed.get("rationale") or ""),
            }
        )
    return observations


def _simple_requires_image_answer(question: str) -> bool:
    """Missing images may veto visible attributes, not textual facts about an entity."""
    text = question.casefold()
    if re.search(r"^(?:in which year|what year|when|where|who|what role|what part|what sports|what was the score|what is the score|what is the release|what is the founding)", text):
        return False
    if re.search(r"\b(colou?r|wearing|holding|visible|eyeglasses|beard|mustache|hair|hand|hands|earrings?|doing|activity)\b", text):
        return True
    has_image = bool(re.search(r"\b(poster|painting|cover|logo|image|photo|picture|emblem|crest)\b", text))
    return has_image and bool(re.search(
        r"how many|\b(?:animal|object|instrument|structure|item|apparel|body part)\b|\b(?:does|is|are)\b.*\b(?:show|feature|contain|there|on|in)\b", text))


def _simple_bound_image_targets(original_question: str, resolved_question: str) -> list[str]:
    """Recover entity bindings from substituted plan references, not visual similarity."""
    if _simple_question_is_visual_selector(resolved_question):
        return []
    pieces = re.split(r"(\{answer_\d+\})", original_question)
    if len(pieces) > 1:
        pattern = "".join("(.+?)" if re.fullmatch(r"\{answer_\d+\}", p) else re.escape(p) for p in pieces)
        match = re.fullmatch(pattern, resolved_question)
        if match:
            targets = []
            for value in match.groups():
                if value.strip().startswith("["):
                    try:
                        values = json.loads(value)
                    except ValueError:
                        try:
                            values = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            values = []
                    # A malformed or non-entity array must never become an image title.
                    if isinstance(values, (list, tuple)):
                        targets.extend(v.strip() for v in values if isinstance(v, str))
                else:
                    targets.append(value.strip())
            return _dedupe_preserve_order([v for v in targets if not _is_unknown_answer(v)])
    match = re.search(r"(?:poster|painting|cover|logo|image)\s+(?:associated with|of|for)\s+(.+?)[?.]?$", resolved_question, re.I)
    if match:
        target = match.group(1).strip()
        if not _simple_requires_image_answer(resolved_question):
            return [target]
        # An unresolved description is not an entity ID (e.g. "the headliners ...").
        if target and target[0].isupper() and not re.search(r"[,?]|\b(?:who|whose|that|which)\b", target):
            return [target]
    return []


async def _simple_visual_verify(
    sub_question: str,
    image_evidence: list[dict],
    funcs: dict,
    enabled: bool = True,
) -> list[dict]:
    vlm_func = funcs.get("vlm_model_func")
    if not enabled or not vlm_func or not image_evidence or not _simple_has_visual_predicate(sub_question):
        return []
    predicate = _simple_visual_predicate(sub_question)
    verifications = []
    for rank, item in enumerate(image_evidence[:4], start=1):
        image_path = item.get("image_path", "")
        if not image_path:
            continue
        candidate = str(item.get("candidate_set_value") or item.get("page_title") or "")
        linked = [str(x) for x in (item.get("linked_entities") or []) if str(x).strip()]
        if linked and (not candidate or candidate.lower() in {"unknown", "image"}):
            candidate = linked[0]
        prompt = PROMPTS["simple_visual_verification_prompt"].format(
            sub_question=predicate,
            candidate=candidate or f"image_rank_{rank}",
            visual_predicate=predicate,
        )
        try:
            response = await vlm_func(image_path=image_path, prompt=prompt, temperature=0.0, max_tokens=96)
            parsed = await parse_json_response(response, funcs["llm_model_func"])
        except Exception:
            parsed = {}
        parsed = parsed if isinstance(parsed, dict) else {}
        result = str(parsed.get("result") or "uncertain").strip().lower()
        if result not in {"yes", "no", "uncertain"}:
            result = "uncertain"
        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0) or 0.0)))
        except Exception:
            confidence = 0.0
        verifications.append(
            {
                "type": "visual_verification",
                "rank": rank,
                "image_id": item.get("image_id", ""),
                "image_path": image_path,
                "page_title": item.get("page_title", ""),
                "linked_entities": linked,
                "candidate_entity": candidate,
                "visual_predicate": predicate,
                "result": result,
                "confidence": confidence,
                "observation": str(parsed.get("observation") or ""),
                "rationale": str(parsed.get("rationale") or ""),
            }
        )
    return verifications


async def _simple_candidate_set_visual_search(
    *,
    sub_question: str,
    node_id: int,
    stores: dict,
    funcs: dict,
    config,
    execution_context: dict,
    limit: int = 12,
) -> tuple[list[dict], list[dict]]:
    if not _simple_question_is_visual_selector(sub_question) or not _simple_has_visual_predicate(sub_question):
        return [], []
    names = _simple_extract_candidate_set_from_question(sub_question, limit=limit)
    if len(names) < 2:
        return [], []
    predicate = _simple_visual_predicate(sub_question)
    merged_images: list[dict] = []
    merged_visuals: list[dict] = []
    seen_images = set()
    for cand_rank, name in enumerate(names[:limit], start=1):
        query = f"{name} {predicate}"
        try:
            image_result = await _simple_image_retrieve(
                query,
                node_id,
                stores,
                funcs,
                config,
                execution_context,
            )
        except Exception:
            continue
        candidate_images = _compact_image_evidence(image_result or {}, limit=4)
        if not candidate_images:
            continue
        matched = next(
            (item for item in candidate_images if _simple_image_matches_candidate_name(name, item)),
            None,
        )
        if not matched:
            merged_visuals.append(
                {
                    "type": "visual_verification",
                    "rank": cand_rank,
                    "candidate_set_value": name,
                    "candidate_set_rank": cand_rank,
                    "visual_predicate": predicate,
                    "result": "uncertain",
                    "confidence": 0.0,
                    "observation": "Skipped because image retrieval returned no candidate-matching image.",
                    "rationale": "candidate_image_mismatch",
                    "retrieved_titles": [
                        item.get("page_title", "")
                        for item in candidate_images[:4]
                    ],
                }
            )
            continue
        image_item = dict(matched)
        image_item["candidate_set_value"] = name
        image_item["candidate_set_rank"] = cand_rank
        image_key = image_item.get("image_id") or image_item.get("image_path") or f"{cand_rank}:{name}"
        if image_key not in seen_images:
            seen_images.add(image_key)
            merged_images.append(image_item)
        checks = await _simple_visual_verify(
            f"Does the image show {predicate}?",
            [image_item],
            funcs,
            enabled=getattr(config, "cor_simple_enable_visual_verifier", True),
        )
        for check in checks:
            check["candidate_set_value"] = name
            check["candidate_set_rank"] = cand_rank
            if str(check.get("result") or "").lower() == "yes":
                check["candidate_entity"] = name
        merged_visuals.extend(checks)
    return merged_images, merged_visuals


async def _simple_disambiguate_visual_selector(
    sub_question: str,
    visual_observations: list[dict],
    funcs: dict,
) -> dict:
    if not _simple_question_is_visual_selector(sub_question):
        return {}
    if re.search(r"\bwhich (?:authors|people|players|teams|institutions)\b|\b(?:wear|have)\b", sub_question, re.I):
        # Set-valued filters may legitimately have several positive candidates.
        return {}
    yes_checks = []
    for obs in visual_observations or []:
        if obs.get("type") != "visual_verification":
            continue
        if str(obs.get("result") or "").lower() != "yes":
            continue
        candidate = str(obs.get("candidate_set_value") or obs.get("candidate_entity") or obs.get("page_title") or "").strip()
        if not candidate:
            continue
        yes_checks.append(
            {
                "candidate": candidate,
                "page_title": obs.get("page_title", ""),
                "confidence": obs.get("confidence", 0.0),
                "observation": obs.get("observation", ""),
                "rationale": obs.get("rationale", ""),
                "candidate_set_value": obs.get("candidate_set_value", ""),
            }
        )
    allowed = _dedupe_preserve_order([item["candidate"] for item in yes_checks])
    if len(allowed) < 2:
        return {}
    prompt = PROMPTS["simple_visual_selector_disambiguation_prompt"].format(
        sub_question=sub_question,
        candidate_checks=json.dumps(yes_checks[:12], ensure_ascii=False),
        allowed_candidates=json.dumps(allowed[:12], ensure_ascii=False),
    )
    try:
        response = await funcs["llm_model_func"](prompt=prompt, temperature=0.0, max_tokens=96)
        parsed = await parse_json_response(response, funcs["llm_model_func"])
    except Exception:
        parsed = {}
    parsed = parsed if isinstance(parsed, dict) else {}
    selected = str(parsed.get("selected_candidate") or "").strip()
    allowed_norm = {_simple_norm_name(value): value for value in allowed}
    selected_norm = _simple_norm_name(selected)
    if selected_norm not in allowed_norm:
        selected = ""
    else:
        selected = allowed_norm[selected_norm]
    event = {
        "type": "visual_selector_disambiguation",
        "selected_candidate": selected,
        "rationale": str(parsed.get("rationale") or ""),
        "allowed_candidates": allowed[:12],
    }
    for obs in visual_observations or []:
        if obs.get("type") != "visual_verification":
            continue
        candidate = str(obs.get("candidate_set_value") or obs.get("candidate_entity") or obs.get("page_title") or "").strip()
        if selected and _simple_norm_name(candidate) == _simple_norm_name(selected):
            obs["selector_selected"] = True
        elif selected:
            obs["selector_selected"] = False
    return event


async def _simple_answer_subquestion(
    *,
    original_query: str,
    sub_question: str,
    previous_summary: str,
    decision: dict,
    text_evidence: list[dict],
    image_evidence: list[dict],
    visual_observations: list[dict],
    funcs: dict,
) -> dict:
    template = PROMPTS["simple_subquestion_answer_prompt"]
    fields = {
        "original_query": _simple_token_truncate(original_query, 500),
        "sub_question": _simple_token_truncate(sub_question, 900),
        "previous_answers": _simple_token_truncate(previous_summary or "None", 1000),
        "chosen_strategy": decision.get("strategy", ""),
    }
    overhead = len(encode_string_by_tiktoken(template.format(**fields, evidence="")))
    evidence_budget = max(0, input_budget(384) - overhead - 32)
    evidence_text = _format_simple_evidence_for_prompt(
        text_evidence, image_evidence, visual_observations, query=sub_question, max_tokens=evidence_budget)
    prompt = template.format(**fields, evidence=evidence_text or "No retrieved evidence.")
    if len(encode_string_by_tiktoken(prompt)) > input_budget(384):
        raise ValueError("Sub-question prompt exceeds the configured Qwen input budget")
    try:
        response = await funcs["llm_model_func"](prompt=prompt, temperature=0.0, max_tokens=384)
        parsed = await parse_json_response(response, funcs["llm_model_func"])
    except Exception as exc:
        parsed = {"answer": "I don't know.", "rationale": f"simple answer failed: {exc}", "confidence": 0.0}
    parsed = parsed if isinstance(parsed, dict) else {}
    value = parsed.get("answer")
    answer = _simple_answer_text(value)
    answer = answer or "I don't know."
    try:
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    return {
        "answer": answer,
        "rationale": str(parsed.get("rationale") or ""),
        "confidence": confidence,
        "failure_type": "" if not _is_unknown_answer(answer) else "unknown_subanswer",
        "packing": {"input_tokens": len(encode_string_by_tiktoken(prompt)), "input_budget": input_budget(384), "evidence_tokens": len(encode_string_by_tiktoken(evidence_text))},
    }


async def _simple_execute_one_attempt(
    *,
    original_query: str,
    node_id: int,
    original_node_question: str,
    resolved_question: str,
    previous_summary: str,
    stores: dict,
    funcs: dict,
    config,
    execution_context: dict,
    allow_direct_answer: bool = True,
) -> tuple[SubQuestionState, dict]:
    if re.search(r"\{answer_\d+\}", resolved_question):
        state = SubQuestionState(
            id=node_id,
            original_question=original_node_question,
            resolved_question=resolved_question,
            chosen_strategy="blocked_unresolved_dependency",
            confidence=0.0,
            failure_type="unresolved_dependency_placeholder",
        )
        return state, {"answer": "I don't know."}

    if getattr(config, "cor_simple_ablation_variant", "full") == "without_sgn":
        # Plain-retrieval ablation: one dense lookup per modality, followed by
        # direct reading of the retrieved text and the original images.
        text_rows = await stores["chunks_vdb"].query(resolved_question, top_k=8)
        chunks = []
        for row in text_rows:
            chunk = await stores["text_chunks_db"].get_by_id(row.get("id") or row.get("__id__"))
            if chunk and chunk.get("content"):
                chunks.append({"content": str(chunk["content"])})
        text_context = pack_text(resolved_question, chunks, int(config.mm_sgn_context_max_tokens))
        text_evidence = [{"type": "context", "content": text_context}] if text_context else []
        image_rows = await stores["images_vdb"].query(
            resolved_question, top_k=int(getattr(config, "image_retrieval_candidate_k", 5)))
        image_evidence = []
        for row in image_rows:
            image_id = row.get("id") or row.get("__id__")
            chunk = await stores["image_chunks_db"].get_by_id(image_id) if image_id else None
            chunk = chunk or {}
            caption = str(chunk.get("caption") or row.get("caption") or "")
            image_evidence.append({
                "image_id": image_id or "", "image_path": chunk.get("image_path") or row.get("image_path") or "",
                "page_title": _page_title_from_caption(caption), "caption": caption,
                "global_description": "", "linked_entities": [],
                "fusion_score": float(row.get("__metrics__", row.get("distance", 0.0)) or 0.0),
            })
        observations = []
        vlm_func = funcs.get("vlm_model_func")
        if vlm_func:
            for image in image_evidence:
                if not image["image_path"]:
                    continue
                prompt = (f"Question: {resolved_question}\n"
                          f"Image page: {image['page_title']}\n"
                          "Inspect this image directly. State only visual facts relevant to the question. "
                          "Do not infer facts that are not visible.")
                try:
                    observation = await vlm_func(image_path=image["image_path"], prompt=prompt,
                                                 temperature=0.0, max_tokens=192)
                except Exception as exc:
                    observation = f"Image reading failed: {exc}"
                observations.append({"image_id": image["image_id"], "page_title": image["page_title"],
                                     "image_path": image["image_path"], "visual_observation": str(observation)})
        decision = {"strategy": "plain_dense_both", "text_query": resolved_question,
                    "image_query": resolved_question}
        sub_answer = await _simple_answer_subquestion(
            original_query=original_query, sub_question=resolved_question,
            previous_summary=previous_summary, decision=decision,
            text_evidence=text_evidence, image_evidence=image_evidence,
            visual_observations=observations, funcs=funcs,
        )
        state = SubQuestionState(
            id=node_id, original_question=original_node_question, resolved_question=resolved_question,
            chosen_strategy="plain_dense_both", confidence=sub_answer.get("confidence", 0.0),
            text_evidence=text_evidence, image_evidence=image_evidence,
            visual_observations=observations, failure_type=sub_answer.get("failure_type", ""),
        )
        return state, {"strategy_decision": decision, "sub_answer": _safe_jsonable(sub_answer),
                       "rationale": sub_answer.get("rationale", ""),
                       "raw_text_result": {"retrieval_backend": "plain_dense_text", "retrieved_count": len(text_rows)},
                       "raw_image_result": {"retrieval_backend": "plain_dense_image", "retrieved_count": len(image_rows)}}

    decision = await _simple_decide_strategy(resolved_question, original_query, previous_summary, funcs)
    strategy = decision["strategy"]
    if not allow_direct_answer and strategy == "direct_answer_if_enough":
        strategy = "both_parallel"
        decision = dict(decision)
        decision["strategy"] = strategy
        decision["rationale"] = "Final verification requires fresh multimodal SGN retrieval."
        decision["text_query"] = resolved_question
        decision["image_query"] = resolved_question
    text_result = None
    image_result = None

    if strategy in {"text_only", "text_then_image", "both_parallel"}:
        text_result = await _simple_text_retrieve(
            decision.get("text_query") or resolved_question, node_id, stores, funcs, config, execution_context
        )
    if strategy in {"image_only", "image_then_text", "both_parallel"}:
        image_result = await _simple_image_retrieve(
            decision.get("image_query") or resolved_question, node_id, stores, funcs, config, execution_context
        )
    if strategy == "text_then_image":
        image_query = "\n".join(
            part
            for part in [
                decision.get("image_query") or resolved_question,
                f"Text answer/context: {(text_result or {}).get('answer', '')}",
            ]
            if part
        )
        image_result = await _simple_image_retrieve(image_query, node_id, stores, funcs, config, execution_context)
    if strategy == "image_then_text":
        image_query_result = image_result or {}
        image_entities = []
        for item in ((image_query_result.get("candidate_state") or {}).get("candidates") or [])[:3]:
            image_entities.append(str(item.get("page_title") or ""))
            image_entities.extend(str(entity) for entity in (item.get("linked_entities") or [])[:2])
        text_query = "\n".join(
            part
            for part in [
                decision.get("text_query") or resolved_question,
                "Image/entity candidates: " + "; ".join(entity for entity in image_entities if entity),
            ]
            if part
        )
        text_result = await _simple_text_retrieve(text_query, node_id, stores, funcs, config, execution_context)

    text_evidence = _compact_text_evidence(text_result or {})
    image_evidence = _compact_image_evidence(image_result or {})
    image_targets = _simple_bound_image_targets(original_node_question, resolved_question)
    if node_id == 900000:
        image_targets = execution_context.get("final_image_targets", image_targets)
    if image_targets:
        image_evidence = [item for item in image_evidence if any(
            _simple_image_matches_candidate_name(target, item) for target in image_targets)]
        if not image_evidence and strategy != "text_only":
            for target in image_targets[:3]:
                retry = await _simple_image_retrieve(target, node_id, stores, funcs, config, execution_context)
                image_evidence.extend(item for item in _compact_image_evidence(retry or {})
                                      if _simple_image_matches_candidate_name(target, item))
    visual_observations = []
    selector_event = {}
    if strategy != "text_only":
        candidate_set_images, candidate_set_visuals = await _simple_candidate_set_visual_search(
            sub_question=resolved_question,
            node_id=node_id,
            stores=stores,
            funcs=funcs,
            config=config,
            execution_context=execution_context,
            limit=12,
        )
        if candidate_set_images:
            existing = {
                item.get("image_id") or item.get("image_path")
                for item in image_evidence
                if item.get("image_id") or item.get("image_path")
            }
            for item in candidate_set_images:
                key = item.get("image_id") or item.get("image_path")
                if key and key in existing:
                    continue
                image_evidence.append(item)
                if key:
                    existing.add(key)
            image_evidence = image_evidence[:8]
        visual_observations.extend(candidate_set_visuals)
        visual_observations.extend(
            await _simple_visual_verify(
                resolved_question,
                image_evidence,
                funcs,
                enabled=getattr(config, "cor_simple_enable_visual_verifier", True),
            )
        )
        selector_event = await _simple_disambiguate_visual_selector(
            resolved_question,
            visual_observations,
            funcs,
        )
        if selector_event:
            visual_observations.append(selector_event)
        visual_observations.extend(await _simple_visual_observe(resolved_question, image_evidence, funcs))
    sub_answer = await _simple_answer_subquestion(
        original_query=original_query,
        sub_question=resolved_question,
        previous_summary=previous_summary,
        decision=decision,
        text_evidence=text_evidence,
        image_evidence=image_evidence,
        visual_observations=visual_observations,
        funcs=funcs,
    )
    if selector_event.get("selected_candidate"):
        sub_answer.update(answer=selector_event["selected_candidate"], failure_type="",
                          rationale=selector_event.get("rationale", ""))
    if image_targets and _simple_requires_image_answer(resolved_question) and strategy != "text_only" and not image_evidence:
        sub_answer.update(answer="I don't know.", confidence=0.0,
                          failure_type="candidate_image_mismatch",
                          rationale="No retrieved image matches the entity bound by the sub-question.")
    state = SubQuestionState(
        id=node_id,
        original_question=original_node_question,
        resolved_question=resolved_question,
        chosen_strategy=strategy,
        confidence=sub_answer.get("confidence", 0.0),
        text_evidence=text_evidence,
        image_evidence=image_evidence,
        visual_observations=visual_observations,
        failure_type=sub_answer.get("failure_type", ""),
    )
    raw = {
        "depends_on": [],
        "strategy_decision": decision,
        "sub_answer": _safe_jsonable(sub_answer),
        "text_answer": (text_result or {}).get("answer", ""),
        "image_answer": (image_result or {}).get("answer", ""),
        "rationale": sub_answer.get("rationale", ""),
        "raw_text_result": _safe_jsonable(text_result or {}),
        "raw_image_result": _safe_jsonable(image_result or {}),
    }
    return state, raw


def _simple_direct_result(state: SubQuestionState, raw_result: dict, node: dict) -> dict:
    """Return the evidence-grounded sub-answer without cross-hop candidate selection."""
    sub_answer = raw_result.get("sub_answer") or {}
    sub_answer = sub_answer if isinstance(sub_answer, dict) else {}
    answer = _simple_answer_text(sub_answer.get("answer")) or "I don't know."
    return {
        "id": int(node.get("id") or state.id),
        "question": str(node.get("question") or state.original_question),
        "resolved_question": state.resolved_question,
        "has_dependency": bool(node.get("depends_on")),
        "depends_on": list(node.get("depends_on") or []),
        "chosen_strategy": state.chosen_strategy,
        "answer": answer,
        "rationale": str(sub_answer.get("rationale") or raw_result.get("rationale") or ""),
        "confidence": float(sub_answer.get("confidence", 0.0) or 0.0),
        "failure_type": "" if not _is_unknown_answer(answer) else (
            state.failure_type or sub_answer.get("failure_type") or "unknown_subanswer"
        ),
        "text_evidence": state.text_evidence,
        "image_evidence": state.image_evidence,
        "visual_observations": state.visual_observations,
        "retrieval_trace": {
            "strategy_decision": raw_result.get("strategy_decision") or {},
            "text_answer": raw_result.get("text_answer", ""),
            "image_answer": raw_result.get("image_answer", ""),
            "packing": sub_answer.get("packing", {}),
        },
    }


def _simple_resolve_dependency_question(
    node: dict,
    sub_results: dict[int, dict],
    original_query: str,
) -> tuple[str, str, bool]:
    resolved = str(node.get("question") or "")
    previous_blocks = []
    has_unknown_dependency = False
    references = {int(ref) for ref in re.findall(r"\{answer_(\d+)\}", resolved)}
    dependencies = sorted(set(node.get("depends_on") or []) | references)
    for dependency_id in dependencies:
        previous = sub_results.get(int(dependency_id)) or {}
        answer = _simple_answer_text(previous.get("answer"))
        unknown = _is_unknown_answer(answer)
        has_unknown_dependency = has_unknown_dependency or unknown
        replacement = f"{{answer_{dependency_id}}}" if unknown else answer
        displayed_answer = answer or "I don't know."
        resolved = resolved.replace(f"{{answer_{dependency_id}}}", replacement)
        resolved = resolved.replace(f"{{{{answer_{dependency_id}}}}}", replacement)
        previous_blocks.append(
            f"Sub-question {dependency_id}: {previous.get('resolved_question') or previous.get('question', '')}\n"
            f"Answer {dependency_id}: {displayed_answer}\n"
            f"Rationale {dependency_id}: {previous.get('rationale', '')}"
        )
    resolved = re.sub(r"\s+", " ", resolved).strip()
    if has_unknown_dependency:
        resolved = f"{original_query}\nFocused sub-question: {resolved}"
    return resolved, "\n\n".join(previous_blocks), has_unknown_dependency


async def _simple_correct_failed_subquestion(
    *,
    original_query: str,
    node: dict,
    failed_result: dict,
    previous_summary: str,
    stores: dict,
    funcs: dict,
    config,
    execution_context: dict,
    planner_model_func,
    depth: int,
    max_depth: int,
) -> tuple[dict, list[dict]]:
    events = []
    if depth >= max_depth:
        return failed_result, events
    prompt = PROMPTS["simple_adaptive_corrector"].format(
        original_query=original_query,
        sub_query=failed_result.get("resolved_question") or node.get("question", ""),
        previous_answers=previous_summary or "None",
        failure_information=json.dumps(
            {
                "failure_type": failed_result.get("failure_type", ""),
                "strategy": failed_result.get("chosen_strategy", ""),
                "text_evidence_count": len(failed_result.get("text_evidence") or []),
                "image_evidence_count": len(failed_result.get("image_evidence") or []),
            },
            ensure_ascii=False,
        ),
    )
    try:
        response = await planner_model_func(prompt=prompt, temperature=0.0, max_tokens=768)
        correction = await parse_json_response(response, planner_model_func)
    except Exception as exc:
        events.append({"event": "adaptive_correction_failed", "node_id": node.get("id"), "error": str(exc)})
        return failed_result, events
    correction = correction if isinstance(correction, dict) else {}
    event = {
        "event": "adaptive_correction_proposed",
        "node_id": node.get("id"),
        "depth": depth,
        "is_single_hop": bool(correction.get("is_single_hop")),
        "rationale": str(correction.get("rationale") or ""),
    }
    events.append(event)

    if correction.get("is_single_hop"):
        rewritten = str(
            correction.get("reconstructed_query")
            or correction.get("reconstructed query")
            or ""
        ).strip()
        if not rewritten or rewritten == failed_result.get("resolved_question"):
            return failed_result, events
        state, raw = await _simple_execute_one_attempt(
            original_query=original_query,
            node_id=int(node.get("id") or 0),
            original_node_question=str(node.get("question") or ""),
            resolved_question=rewritten,
            previous_summary=previous_summary,
            stores=stores,
            funcs=funcs,
            config=config,
            execution_context=execution_context,
        )
        retry = _simple_direct_result(state, raw, node)
        retry["corrected_from"] = failed_result.get("resolved_question", "")
        retry["correction_type"] = "rewrite_and_retry"
        events.append(
            {
                "event": "adaptive_correction_executed",
                "node_id": node.get("id"),
                "correction_type": "rewrite_and_retry",
                "rewritten_query": rewritten,
                "answer": retry.get("answer"),
            }
        )
        return retry if not _is_unknown_answer(retry.get("answer")) else failed_result, events

    correction_subqueries = correction.get("sub_queries") or []
    if not correction_subqueries:
        try:
            redecompose_response = await planner_model_func(
                prompt=PROMPTS["simple_question_decomposition"].format(
                    query=failed_result.get("resolved_question") or node.get("question", "")
                ),
                temperature=0.0,
                max_tokens=768,
            )
            redecomposed = await parse_json_response(redecompose_response, planner_model_func)
            redecomposed = redecomposed if isinstance(redecomposed, dict) else {}
            correction_subqueries = redecomposed.get("sub_queries") or redecomposed.get("sub_questions") or []
            events.append(
                {
                    "event": "adaptive_correction_redecomposed",
                    "node_id": node.get("id"),
                    "num_subquestions": len(correction_subqueries),
                }
            )
        except Exception as exc:
            events.append(
                {
                    "event": "adaptive_correction_redecomposition_failed",
                    "node_id": node.get("id"),
                    "error": str(exc),
                }
            )
    nested_plan = _normalize_simple_plan(
        {"reasoning_plan": correction.get("rationale", ""), "sub_queries": correction_subqueries},
        failed_result.get("resolved_question") or node.get("question", ""),
        max_nodes=6,
    )
    nested_results, nested_events = await _run_simple_natural_leg(
        original_query=failed_result.get("resolved_question") or node.get("question", ""),
        plan=nested_plan,
        stores=stores,
        funcs=funcs,
        config=config,
        execution_context=execution_context,
        planner_model_func=planner_model_func,
        depth=depth + 1,
        max_depth=max_depth,
    )
    events.extend(nested_events)
    nested_summary = _simple_results_summary(nested_results)
    if not nested_summary:
        return failed_result, events
    retry_question = failed_result.get("resolved_question") or node.get("question", "")
    state, raw = await _simple_execute_one_attempt(
        original_query=original_query,
        node_id=int(node.get("id") or 0),
        original_node_question=str(node.get("question") or ""),
        resolved_question=retry_question,
        previous_summary="\n\n".join(part for part in [previous_summary, nested_summary] if part),
        stores=stores,
        funcs=funcs,
        config=config,
        execution_context=execution_context,
    )
    retry = _simple_direct_result(state, raw, node)
    retry["correction_type"] = "recursive_decomposition"
    retry["correction_subquestions"] = nested_results
    events.append(
        {
            "event": "adaptive_correction_executed",
            "node_id": node.get("id"),
            "correction_type": "recursive_decomposition",
            "answer": retry.get("answer"),
        }
    )
    return retry if not _is_unknown_answer(retry.get("answer")) else failed_result, events


def _simple_final_image_targets(results: list[dict]) -> list[str]:
    # A substituted entity scopes its own hop, not the original question.
    # Only an explicit visual selector winner can constrain final verification.
    return _dedupe_preserve_order([
        observation["selected_candidate"]
        for result in results
        if not result.get("failure_type") and not _is_unknown_answer(result.get("answer"))
        for observation in result.get("visual_observations", [])
        if observation.get("type") == "visual_selector_disambiguation"
        and not _is_unknown_answer(observation.get("selected_candidate"))
    ])


def _simple_results_summary(results: list[dict]) -> str:
    return "\n\n".join(
        f"Sub-question {item.get('id')}: {item.get('resolved_question') or item.get('question', '')}\n"
        f"Answer {item.get('id')}: {item.get('answer', '')}\n"
        f"Rationale {item.get('id')}: {item.get('rationale', '')}"
        for item in results
        if not _is_unknown_answer(item.get("answer"))
    )


def _simple_token_truncate(value: object, max_tokens: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    tokens = encode_string_by_tiktoken(text)
    if len(tokens) <= max_tokens:
        return text
    return decode_tokens_by_tiktoken(tokens[:max(0, max_tokens)])


def _pack_final_subquestion_answers(results: list[dict], max_tokens: int = 2200) -> str:
    """Budget each answer, so long early evidence cannot erase later steps."""
    per_result = max(32, (max_tokens - 32) // max(1, len(results)))
    packed = []
    for result in results:
        packed.append({
            "id": result.get("id"),
            "question": _simple_token_truncate(result.get("resolved_question") or result.get("question", ""), per_result // 4),
            "answer": _simple_token_truncate(result.get("answer", ""), per_result // 2),
            "rationale": _simple_token_truncate(result.get("rationale", ""), per_result // 8),
        })
    return json.dumps(packed, ensure_ascii=False)


def _simple_result_for_final(result: dict, *, text_limit: int = 3, image_limit: int = 3) -> dict:
    return {
        "id": result.get("id"),
        "question": result.get("resolved_question") or result.get("question", ""),
        "answer": result.get("answer", ""),
        "rationale": str(result.get("rationale") or "")[:800],
        "strategy": result.get("chosen_strategy", ""),
        "text_evidence": [
            {"type": item.get("type", "context"), "content": pack_text(result.get("resolved_question") or result.get("question", ""), [item], 350)}
            for item in (result.get("text_evidence") or [])[:text_limit]
        ],
        "image_evidence": [
            {
                "image_id": item.get("image_id", ""),
                "page_title": item.get("page_title", ""),
                "linked_entities": (item.get("linked_entities") or [])[:3],
                "caption": str(item.get("caption") or "")[:350],
                "global_description": str(item.get("global_description") or "")[:350],
            }
            for item in (result.get("image_evidence") or [])[:image_limit]
        ],
        "visual_observations": _compact_visual_observations(result.get("visual_observations") or []),
    }


async def _run_simple_natural_leg(
    *,
    original_query: str,
    plan: dict,
    stores: dict,
    funcs: dict,
    config,
    execution_context: dict,
    planner_model_func,
    depth: int = 0,
    max_depth: int = 1,
) -> tuple[list[dict], list[dict]]:
    results: dict[int, dict] = {}
    trace = []
    executed: set[int] = set()
    while len(executed) < len(plan.get("sub_questions") or []):
        ready = _simple_ready_nodes(plan, executed)
        if not ready:
            trace.append({"event": "simple_leg_blocked", "depth": depth, "reason": "No sub-question was ready."})
            break
        node = ready[0]
        node_id = int(node["id"])
        resolved, previous_summary, had_unknown_dependency = _simple_resolve_dependency_question(
            node, results, original_query
        )
        state, raw = await _simple_execute_one_attempt(
            original_query=original_query,
            node_id=(depth * 1000) + node_id,
            original_node_question=node.get("question", ""),
            resolved_question=resolved,
            previous_summary=previous_summary,
            stores=stores,
            funcs=funcs,
            config=config,
            execution_context=execution_context,
        )
        result = _simple_direct_result(state, raw, node)
        result["dependency_recovery_used"] = had_unknown_dependency
        if _is_unknown_answer(result.get("answer")):
            corrected, correction_events = await _simple_correct_failed_subquestion(
                original_query=original_query,
                node=node,
                failed_result=result,
                previous_summary=previous_summary,
                stores=stores,
                funcs=funcs,
                config=config,
                execution_context=execution_context,
                planner_model_func=planner_model_func,
                depth=depth,
                max_depth=max_depth,
            )
            result = corrected
            trace.extend(correction_events)
        results[node_id] = result
        executed.add(node_id)
        trace.append(
            {
                "event": "simple_leg_node_executed",
                "depth": depth,
                "node_id": node_id,
                "question": node.get("question", ""),
                "resolved_question": result.get("resolved_question", ""),
                "depends_on": node.get("depends_on") or [],
                "chosen_strategy": result.get("chosen_strategy", ""),
                "answer": result.get("answer", ""),
                "failure_type": result.get("failure_type", ""),
            }
        )
    return [results[idx] for idx in sorted(results)], trace


async def cor_simple_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    global_config: dict,
    images_vdb: BaseVectorStorage = None,
    image_text_vdb: BaseVectorStorage = None,
    image_chunks_db: BaseKVStorage = None,
    visual_entities_db: BaseKVStorage = None,
    query_image_path: str = None,
) -> dict:
    """Natural-language text CoRProRAG execution with multimodal SGN tools."""
    use_model_func = global_config["llm_model_func"]
    planner_model_func = global_config.get("planner_model_func") or use_model_func
    config = global_config["config"]
    variant = getattr(config, "cor_simple_ablation_variant", "full")
    if variant not in SIMPLE_ABLATION_VARIANTS:
        raise ValueError(f"Unknown cor_simple ablation variant: {variant}")
    replay_path = str(getattr(config, "cor_simple_plan_replay_path", "") or "").strip()
    if variant == "sgn_only" and replay_path:
        raise ValueError("SGN-only cannot use a decomposition replay")
    stores = {
        "knowledge_graph_inst": knowledge_graph_inst,
        "entities_vdb": entities_vdb,
        "relationships_vdb": relationships_vdb,
        "chunks_vdb": chunks_vdb,
        "text_chunks_db": text_chunks_db,
        "images_vdb": images_vdb,
        "image_text_vdb": image_text_vdb,
        "image_chunks_db": image_chunks_db,
        "visual_entities_db": visual_entities_db,
    }
    funcs = {
        "llm_model_func": use_model_func,
        "vlm_model_func": global_config.get("vlm_model_func"),
        "rerank_texts_func": global_config.get("rerank_texts_func"),
    }
    execution_context = {"original_query": query, "query_image_path": query_image_path, "node_results": {}}
    if variant == "sgn_only":
        plan = {"reasoning_plan": "Single-pass SGN on the original question.", "sub_questions": []}
        planner_error = ""
        max_depth = 0
        subquestion_results, execution_trace = [], []
    else:
        if replay_path:
            saved_plan = _simple_load_plan_replay(replay_path).get(query)
            if saved_plan is None:
                raise ValueError("Question is absent from the decomposition replay")
            parsed_plan = json.loads(json.dumps(saved_plan, ensure_ascii=False))
            planner_error = ""
        else:
            try:
                response = await planner_model_func(
                    prompt=PROMPTS["simple_question_decomposition"].format(query=query),
                    temperature=0.0,
                    max_tokens=1024,
                )
                parsed_plan = await parse_json_response(response, planner_model_func)
            except Exception as exc:
                parsed_plan = {}
                planner_error = str(exc)
            else:
                planner_error = ""
        plan = _normalize_simple_plan(parsed_plan, query, max_nodes=6)
        max_depth = 0 if variant == "without_ac" else max(0, int(getattr(config, "cor_simple_max_correction_depth", 1)))
        subquestion_results, execution_trace = await _run_simple_natural_leg(
            original_query=query,
            plan=plan,
            stores=stores,
            funcs=funcs,
            config=config,
            execution_context=execution_context,
            planner_model_func=planner_model_func,
            max_depth=max_depth,
        )

    previous_summary = _simple_results_summary(subquestion_results)
    execution_context["final_image_targets"] = _simple_final_image_targets(subquestion_results)
    final_state, final_raw = await _simple_execute_one_attempt(
        original_query=query,
        node_id=900000,
        original_node_question=query,
        resolved_question=query,
        previous_summary=previous_summary,
        stores=stores,
        funcs=funcs,
        config=config,
        execution_context=execution_context,
        allow_direct_answer=False,
    )
    final_retrieval = _simple_direct_result(
        final_state,
        final_raw,
        {"id": 900000, "question": query, "depends_on": []},
    )
    final_output_tokens = max(96, int(getattr(config, "cor_simple_final_output_tokens", 256)))
    final_input_budget = min(input_budget(final_output_tokens), max(3000, int(getattr(config, "cor_simple_final_input_tokens", 6200))))
    plan_for_final = _simple_token_truncate(plan, 900)
    compact_subresults = _pack_final_subquestion_answers(subquestion_results)
    final_retrieval_payload = _simple_result_for_final(final_retrieval, text_limit=4, image_limit=4)
    prompt_without_final_evidence = PROMPTS["simple_cor_final_answer_prompt"].format(
        original_query=query,
        simple_plan=plan_for_final,
        subquestion_results=compact_subresults,
        final_retrieval="",
    )
    final_evidence_budget = max(
        256,
        final_input_budget - len(encode_string_by_tiktoken(prompt_without_final_evidence)) - 128,
    )
    compact_final_retrieval = _simple_token_truncate(final_retrieval_payload, final_evidence_budget)
    final_prompt = PROMPTS["simple_cor_final_answer_prompt"].format(
        original_query=query,
        simple_plan=plan_for_final,
        subquestion_results=compact_subresults,
        final_retrieval=compact_final_retrieval,
    )
    final_prompt_tokens = encode_string_by_tiktoken(final_prompt)
    if len(final_prompt_tokens) > final_input_budget:
        overflow = len(final_prompt_tokens) - final_input_budget
        compact_final_retrieval = _simple_token_truncate(
            compact_final_retrieval,
            max(128, final_evidence_budget - overflow - 64),
        )
        final_prompt = PROMPTS["simple_cor_final_answer_prompt"].format(
            original_query=query,
            simple_plan=plan_for_final,
            subquestion_results=compact_subresults,
            final_retrieval=compact_final_retrieval,
        )
    try:
        if len(encode_string_by_tiktoken(final_prompt)) > final_input_budget:
            raise ValueError("Final synthesis prompt exceeds the configured Qwen input budget")
        final_response = await use_model_func(
            prompt=final_prompt,
            temperature=0.0,
            max_tokens=final_output_tokens,
        )
        final_output = await parse_json_response(final_response, use_model_func)
    except Exception as exc:
        final_output = {"answer": "I don't know.", "rationale": f"Final synthesis failed: {exc}"}
    final_output = final_output if isinstance(final_output, dict) else {}
    if isinstance(final_output.get("answer"), (bool, int, float)):
        final_output["answer"] = _simple_answer_text(final_output["answer"])
    if _is_unknown_answer(final_output.get("answer")):
        fallback = final_retrieval.get("answer")
        # Only the retrieval for the original question can provide a fallback.
        # An intermediate movie/person/number is not the requested final answer.
        final_output["answer"] = "I don't know." if _is_unknown_answer(fallback) else fallback
    final_output.setdefault("rationale", "")
    final_output.update(
        {
            "method_variant": "cor_simple_text_aligned_multimodal",
            "ablation_variant": variant,
            "plan_source": "replay" if replay_path else ("none" if variant == "sgn_only" else "planner"),
            "simple_planner_model": "" if variant == "sgn_only" else (
                getattr(config, "simple_planner_model_name", "") or getattr(config, "llm_model_name", "")),
            "simple_plan": plan,
            "subquestion_results": subquestion_results,
            "final_sgn_retrieval": final_retrieval,
            "final_packing": {
                "input_tokens": len(encode_string_by_tiktoken(final_prompt)),
                "input_budget": final_input_budget,
                "output_tokens": final_output_tokens,
            },
            "execution_trace": ([{"event": "planner_fallback", "error": planner_error}] if planner_error else [])
            + execution_trace
            + [{"event": "final_original_question_sgn", "answer": final_retrieval.get("answer"), "strategy": final_retrieval.get("chosen_strategy")}],
            "adaptive_corrector": {"enabled": max_depth > 0, "max_depth": max_depth},
            "logic_harness": {"enabled": False},
        }
    )
    return final_output
