import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.pipeline.query_executor import (
    cor_simple_query,
    _is_unknown_answer,
    _simple_answer_text,
    _simple_final_image_targets,
    _normalize_simple_plan,
    _simple_correct_failed_subquestion,
    _simple_resolve_dependency_question,
    _format_simple_evidence_for_prompt,
    _pack_final_subquestion_answers,
    _simple_decide_strategy,
    _simple_extract_candidate_set_from_question,
    _simple_visual_predicate,
    _simple_disambiguate_visual_selector,
    _split_candidate_answer,
    _simple_execute_one_attempt,
    _simple_bound_image_targets,
    _simple_image_matches_candidate_name,
    _simple_answer_subquestion,
)
from src.pipeline.context_packing import encode, pack_text, input_budget
from src.pipeline.simple_state import SubQuestionState
from testCode.Evaluation.GenerateAnswers import normalize_method, score_answer, write_eval_result_to_csv, process_queries


class CorSimpleAlignmentTest(unittest.IsolatedAsyncioTestCase):
    async def test_json_zero_and_false_survive_reader_and_dependency(self):
        for value, expected in [(0, "0"), (False, "no"), (True, "yes")]:
            self.assertFalse(_is_unknown_answer(value))
            result = await _simple_answer_subquestion(
                original_query="Question", sub_question="Question", previous_summary="",
                decision={"strategy": "text_only"}, text_evidence=[], image_evidence=[],
                visual_observations=[], funcs={"llm_model_func": AsyncMock(
                    return_value=json.dumps({"answer": value}))})
            self.assertEqual(result["answer"], expected)
            self.assertEqual(result["failure_type"], "")
            question, _, blocked = _simple_resolve_dependency_question(
                {"question": "Use {answer_1}", "depends_on": [1]},
                {1: {"answer": value}}, "Original")
            self.assertEqual(question, "Use " + expected)
            self.assertFalse(blocked)
        self.assertEqual(_simple_answer_text("false"), "false")
        self.assertEqual(_simple_answer_text(["Hello, Dolly!"]), '["Hello, Dolly!"]')

    def test_empty_structured_answers_do_not_enter_next_hop(self):
        for answer in [[], {}, "[]", "[ ]", "{}", "null", "unknown"]:
            self.assertTrue(_is_unknown_answer(answer))
            question, _, blocked = _simple_resolve_dependency_question(
                {"question": "Who is the artist of {answer_1}?", "depends_on": [1]},
                {1: {"answer": answer}}, "Original question")
            self.assertTrue(blocked)
            self.assertIn("{answer_1}", question)
        for answer in ['["Gifted"]', 'No', '0', 'two']:
            self.assertFalse(_is_unknown_answer(answer))

    def test_final_retrieval_does_not_inherit_unverified_hop_binding(self):
        result = {"question": "What car is on the poster of {answer_1}?",
                  "resolved_question": 'What car is on the poster of ["Gifted"]?',
                  "chosen_strategy": "text_then_image", "answer": "Red car"}
        self.assertTrue(_simple_bound_image_targets(result["question"], result["resolved_question"]))
        self.assertEqual(_simple_final_image_targets([result]), [])
        winner = {"answer": "Redwing", "visual_observations": [
            {"type": "visual_selector_disambiguation", "selected_candidate": "Redwing"}]}
        self.assertEqual(_simple_final_image_targets([result, winner]), ["Redwing"])
        winner["answer"] = "[]"
        self.assertEqual(_simple_final_image_targets([winner]), [])

    def test_binding_expands_array_without_splitting_title_commas(self):
        original = "What color is the poster of {answer_1}?"
        resolved = 'What color is the poster of ["In the Heights", "Hello, Dolly!"]?'
        self.assertEqual(_simple_bound_image_targets(original, resolved), ["In the Heights", "Hello, Dolly!"])

    def test_description_is_not_used_as_an_image_identity(self):
        question = "In the image of the headliners of the Get a Grip Tour, how many members hold a guitar?"
        self.assertEqual(_simple_bound_image_targets(question, question), [])

    async def test_missing_image_cannot_erase_textual_answer(self):
        for original, question, answer in [
            ("In which year did MC Chris feature in {answer_1}?", 'In which year did MC Chris feature in ["The Chris Gethard Show", "Penn & Teller"]?', "2011"),
            ("What role did Jonny Weston play in {answer_1}?", 'What role did Jonny Weston play in ["About Cherry", "Stag"]?', "Bobby"),
            ("What sports are associated with {answer_1}?", 'What sports are associated with ["The Bad News Bears", "Carrie"]?', "baseball"),
        ]:
            with patch("src.pipeline.query_executor._simple_decide_strategy", new=AsyncMock(return_value={"strategy":"text_then_image"})), \
                 patch("src.pipeline.query_executor._simple_text_retrieve", new=AsyncMock(return_value={"answer":answer,"evidence":[{"content":answer}]})), \
                 patch("src.pipeline.query_executor._simple_image_retrieve", new=AsyncMock(return_value={})), \
                 patch("src.pipeline.query_executor._simple_candidate_set_visual_search", new=AsyncMock(return_value=([],[]))), \
                 patch("src.pipeline.query_executor._simple_visual_verify", new=AsyncMock(return_value=[])), \
                 patch("src.pipeline.query_executor._simple_visual_observe", new=AsyncMock(return_value=[])), \
                 patch("src.pipeline.query_executor._simple_answer_subquestion", new=AsyncMock(return_value={"answer":answer})):
                state, raw = await _simple_execute_one_attempt(original_query=question, node_id=2,
                    original_node_question=original, resolved_question=question, previous_summary="",
                    stores={}, funcs={}, config=SimpleNamespace(), execution_context={})
            self.assertEqual(raw["sub_answer"]["answer"], answer)
            self.assertNotEqual(state.failure_type, "candidate_image_mismatch")

    async def test_runtime_failure_stays_in_results_and_checkpoint(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / "answers.json")
            with patch("testCode.Evaluation.GenerateAnswers.fetch_single_query", new=AsyncMock(return_value={"index":0,"error":"service unavailable"})):
                result = await process_queries(None, [{"question":"Which bird?", "answer":["Redwing"]}], out, str(Path(directory)/"eval.csv"), concurrency=1)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["answer"]["execution_error"], "service unavailable")
            saved = json.loads(Path(out + ".partial.jsonl").read_text())
            self.assertEqual(saved["traditional_metrics"]["EM"], 0)
            self.assertEqual(saved["index"], 0)

    def test_image_identity_is_not_substring_matching(self):
        self.assertFalse(_simple_image_matches_candidate_name("Ali", {"page_title": "Alien"}))
        self.assertTrue(_simple_image_matches_candidate_name("The Polar Express", {"page_title": "The Polar Express (film)"}))

    def test_mmqa_requires_all_gold_answers(self):
        gold = ["Richard Powers", "Joshua Ferris", "David Nicholls", "Karen Joy Fowler"]
        meta = {"dataset": "multimodalqa"}
        em, f1, precision, recall = score_answer('["Joshua Ferris"]', gold, meta)
        self.assertEqual((em, f1, precision, recall), (0, 0.4, 1.0, 0.25))
        self.assertEqual(score_answer(json.dumps(gold[::-1]), gold, meta), (1, 1.0, 1.0, 1.0))
        self.assertEqual(score_answer("Joshua Ferris", gold, {"dataset": "other"})[0], 1)

    def test_csv_quotes_roundtrip(self):
        import tempfile, csv
        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            row = dict(query='Which "film", first?', prediction='["a", "b"]', ground_truth=['a','b'],
                       EM=1, F1=1, Precision=1, Recall=1, M3VQA_Accuracy='')
            write_eval_result_to_csv(f.name, row, "Skipped")
            with open(f.name) as inp:
                values = next(csv.reader(inp))
            self.assertEqual(len(values), 9)
            self.assertEqual(values[1], row['prediction'])

    async def test_selector_winner_becomes_node_answer(self):
        image = {"image_id": "redwing", "page_title": "Redwing", "image_path": "bird.jpg"}
        with patch("src.pipeline.query_executor._simple_decide_strategy", new=AsyncMock(return_value={"strategy":"image_only"})), \
             patch("src.pipeline.query_executor._simple_image_retrieve", new=AsyncMock(return_value={})), \
             patch("src.pipeline.query_executor._simple_candidate_set_visual_search", new=AsyncMock(return_value=([image], []))), \
             patch("src.pipeline.query_executor._simple_visual_verify", new=AsyncMock(return_value=[])), \
             patch("src.pipeline.query_executor._simple_visual_observe", new=AsyncMock(return_value=[])), \
             patch("src.pipeline.query_executor._simple_disambiguate_visual_selector", new=AsyncMock(return_value={"selected_candidate":"Redwing"})), \
             patch("src.pipeline.query_executor._simple_answer_subquestion", new=AsyncMock(return_value={"answer":'["Redwing", "Dusky thrush"]'})):
            state, raw = await _simple_execute_one_attempt(original_query="Which bird?", node_id=2,
                original_node_question="Which of {answer_1} has eye stripes?",
                resolved_question="Which of Redwing, Dusky thrush has eye stripes?", previous_summary="",
                stores={}, funcs={}, config=SimpleNamespace(), execution_context={})
        self.assertEqual(raw["sub_answer"]["answer"], "Redwing")

    async def test_other_movie_image_cannot_answer_bound_painting(self):
        question = "How many characters are in the painting associated with Twelfth Night?"
        self.assertEqual(_simple_bound_image_targets("How many characters are in the painting associated with {answer_1}?", question), ["Twelfth Night"])
        image = {"candidate_state":{"candidates":[{"image_id":"wrong", "page_title":"A Chinese Ghost Story", "image_path":"wrong.jpg"}]}}
        observe = AsyncMock(return_value=[])
        with patch("src.pipeline.query_executor._simple_decide_strategy", new=AsyncMock(return_value={"strategy":"image_only"})), \
             patch("src.pipeline.query_executor._simple_image_retrieve", new=AsyncMock(return_value=image)), \
             patch("src.pipeline.query_executor._simple_candidate_set_visual_search", new=AsyncMock(return_value=([], []))), \
             patch("src.pipeline.query_executor._simple_visual_verify", new=AsyncMock(return_value=[])), \
             patch("src.pipeline.query_executor._simple_visual_observe", new=observe), \
             patch("src.pipeline.query_executor._simple_answer_subquestion", new=AsyncMock(return_value={"answer":"5"})):
            state, raw = await _simple_execute_one_attempt(original_query=question, node_id=2,
                original_node_question="How many characters are in the painting associated with {answer_1}?",
                resolved_question=question, previous_summary="", stores={}, funcs={},
                config=SimpleNamespace(), execution_context={})
        self.assertEqual(observe.await_args.args[1], [])
        self.assertEqual(raw["sub_answer"]["answer"], "I don't know.")
        self.assertEqual(state.failure_type, "candidate_image_mismatch")

    async def test_simple_pipeline_executes_text_then_visual_hop(self):
        planner = AsyncMock(return_value=json.dumps({"sub_queries": [
            {"query": "Which film credits the actor?"},
            {"query": "What is the person holding in the poster of {answer_1}?", "depends_on": [1]},
        ]}))
        llm = AsyncMock(return_value=json.dumps({"answer": "wand", "rationale": "Seen in the poster."}))
        vlm = AsyncMock(return_value=json.dumps({"result": "yes", "confidence": 0.9,
                                               "visual_observation": "A wand", "observation": "A wand"}))
        image_result = {"candidate_state": {"candidates": [{"image_id": "poster",
                        "image_path": "mock-poster.jpg", "page_title": "Example Film"}]}}
        with patch("src.pipeline.query_executor._simple_decide_strategy", new=AsyncMock(side_effect=[
            {"strategy": "text_only"}, {"strategy": "image_only"}, {"strategy": "both_parallel"},
        ])), patch("src.pipeline.query_executor._simple_text_retrieve", new=AsyncMock(return_value={
            "answer": "Example Film", "evidence": [{"content": "Actor credited in Example Film"}],
        })), patch("src.pipeline.query_executor._simple_image_retrieve", new=AsyncMock(return_value=image_result)), \
             patch("src.pipeline.query_executor._simple_answer_subquestion", new=AsyncMock(side_effect=[
                 {"answer": "Example Film", "confidence": 0.9},
                 {"answer": "wand", "confidence": 0.9},
                 {"answer": "wand", "confidence": 0.9},
             ])):
            result = await cor_simple_query(
                query="What does the actor hold in the poster?", knowledge_graph_inst=None,
                entities_vdb=None, relationships_vdb=None, chunks_vdb=None, text_chunks_db=None,
                global_config={"llm_model_func": llm, "planner_model_func": planner,
                               "vlm_model_func": vlm, "config": SimpleNamespace()},
            )
        self.assertEqual(result["answer"], "wand")
        self.assertIn("Example Film", result["subquestion_results"][1]["resolved_question"])
        self.assertEqual(result["subquestion_results"][1]["image_evidence"][0]["image_id"], "poster")
        self.assertGreater(vlm.await_count, 0)

    def test_comparison_references_are_inferred_in_addition_to_operands(self):
        plan = _normalize_simple_plan({"sub_questions": [
            {"question": "Which film?"},
            {"question": "Which other film?"},
            {"question": "Year of {answer_1}?", "depends_on": [1]},
            {"question": "Year of {answer_2}?", "depends_on": [2]},
            {"question": "Which came first, {answer_1} or {answer_2}?", "depends_on": [3, 4]},
        ]}, "Compare films")
        node = plan["sub_questions"][-1]
        self.assertEqual(node["depends_on"], [1, 2, 3, 4])
        # Also accept a pre-existing cached plan with the incomplete dependency list.
        node["depends_on"] = [3, 4]
        resolved, _, _ = _simple_resolve_dependency_question(node, {
            1: {"answer": "Film A"}, 2: {"answer": "Film B"},
            3: {"answer": "1990"}, 4: {"answer": "2007"},
        }, "Compare films")
        self.assertEqual(resolved, "Which came first, Film A or Film B?")

    def test_late_visual_evidence_reaches_reader_even_after_long_text(self):
        observations = [{"type": "visual_verification", "candidate_entity": f"Bird {i}",
                         "result": "no", "observation": "No eye stripes"} for i in range(9)]
        observations.append({"type": "visual_verification", "candidate_entity": "Redwing",
                             "result": "yes", "observation": "Stripes above and below eye"})
        prompt = _format_simple_evidence_for_prompt(
            [{"content": "irrelevant text " * 1000} for _ in range(8)], [], observations
        )
        self.assertIn('"candidate": "Redwing"', prompt)
        self.assertIn('"result": "yes"', prompt)
        self.assertIn('"candidate": "Bird 0"', prompt)
        self.assertLessEqual(len(encode(prompt)), 4000)

    def test_structured_candidates_keep_commas_and_apostrophes(self):
        names = ["Redwing", "O'Brien", "Hello, Dolly!"]
        for serialized in (json.dumps(names), str(names)):
            self.assertEqual(_split_candidate_answer(serialized, "which of candidate list"), names)

    def test_bird_query_contains_only_current_candidate_and_predicate(self):
        question = "Which bird from ['Redwing', 'Dusky thrush', 'Fieldfare'] has a stripe above and below its eye?"
        self.assertEqual(_simple_extract_candidate_set_from_question(question), ["Redwing", "Dusky thrush", "Fieldfare"])
        predicate = _simple_visual_predicate(question)
        self.assertEqual(predicate, "a stripe above and below its eye")
        self.assertNotIn("Dusky", f"Redwing {predicate}")

    async def test_late_table_role_survives_with_real_qwen_prompt_budget(self):
        table = "Nona Gaye filmography\nColumns: Year | Title | Role\n" + "\n".join(
            f"Row {i}: 2000 | Unrelated Film {i} | Other" for i in range(200))
        table += "\nRow 201: 2004 | The Polar Express | Hero Girl (voice)"
        question = "What role did Nona Gaye play in The Polar Express?"
        packed = pack_text(question, [{"content": table}], 250)
        self.assertIn("Hero Girl (voice)", packed)
        self.assertIn("Columns: Year | Title | Role", packed)
        self.assertLessEqual(len(encode(packed)), 250)
        llm = AsyncMock(return_value=json.dumps({"answer": "Hero Girl (voice)"}))
        result = await _simple_answer_subquestion(
            original_query=question, sub_question=question, previous_summary="The Polar Express",
            decision={"strategy": "text_only"}, text_evidence=[{"content": table}],
            image_evidence=[], visual_observations=[], funcs={"llm_model_func": llm})
        prompt = llm.await_args.kwargs["prompt"]
        self.assertIn("Hero Girl (voice)", prompt)
        self.assertLessEqual(len(encode(prompt)), input_budget(384))
        self.assertEqual(result["packing"]["input_tokens"], len(encode(prompt)))

    def test_final_packing_preserves_terminal_answer_after_long_early_evidence(self):
        results = [{"id": i, "question": "A question", "answer": "long candidate " * 500,
                    "text_evidence": [{"content": "long source " * 5000}]} for i in range(1, 6)]
        results.append({"id": 6, "question": "What is the final answer?", "answer": "pink"})
        packed = json.loads(_pack_final_subquestion_answers(results))
        self.assertEqual(len(packed), 6)
        self.assertEqual(packed[-1]["answer"], "pink")

    def test_eyeglasses_filter_extracts_all_candidates(self):
        question = "Which of Joshua Ferris, David Nicholls, Karen Joy Fowler wear eyeglasses?"
        self.assertEqual(_simple_extract_candidate_set_from_question(question),
                         ["Joshua Ferris", "David Nicholls", "Karen Joy Fowler"])
        self.assertIn("eyeglasses", _simple_visual_predicate(question))

    async def test_plural_visual_filter_does_not_force_one_winner(self):
        llm = AsyncMock()
        event = await _simple_disambiguate_visual_selector(
            "Which of Joshua Ferris, David Nicholls wear eyeglasses?",
            [{"type": "visual_verification", "candidate_entity": name, "result": "yes"}
             for name in ["Joshua Ferris", "David Nicholls"]], {"llm_model_func": llm})
        self.assertEqual(event, {})
        llm.assert_not_awaited()

    async def test_visible_object_cannot_be_answered_from_film_title(self):
        llm = AsyncMock(return_value=json.dumps({"strategy": "text_only", "text_query": "Goblet of Fire"}))
        question = "In the title of Harry Potter and the Goblet of Fire, what is Harry holding in his hand?"
        result = await _simple_decide_strategy(question, question, "", {"llm_model_func": llm})
        self.assertEqual(result["strategy"], "image_only")
        self.assertEqual(result["image_query"], question)

    async def test_failed_final_does_not_return_intermediate_movie_as_color(self):
        llm = AsyncMock(return_value=json.dumps({"answer": "I don't know.", "rationale": "Color not established."}))
        planner = AsyncMock(return_value=json.dumps({"sub_questions": [{"question": "Which film?"}]}))
        final_state = SubQuestionState(900000, "What color?", "What color?", "image_only")
        final_raw = {"sub_answer": {"answer": "I don't know.", "confidence": 0.0}}
        steps = [{"id": 1, "question": "Which film?", "answer": "The Pilot"},
                 {"id": 2, "question": "What color?", "answer": "I don't know."}]
        with patch("src.pipeline.query_executor._run_simple_natural_leg", new=AsyncMock(return_value=(steps, []))), \
             patch("src.pipeline.query_executor._simple_execute_one_attempt", new=AsyncMock(return_value=(final_state, final_raw))):
            result = await cor_simple_query(
                query="What color?", knowledge_graph_inst=None, entities_vdb=None,
                relationships_vdb=None, chunks_vdb=None, text_chunks_db=None,
                global_config={"llm_model_func": llm, "planner_model_func": planner, "config": SimpleNamespace()},
            )
        self.assertEqual(result["answer"], "I don't know.")

    def test_original_text_cor_plan_schema_is_supported(self):
        plan = _normalize_simple_plan(
            {
                "reasoning_plan": "Resolve the entity and then its attribute.",
                "sub_queries": [
                    {"query": "Who played Muriel?", "has_dependency": False},
                    {
                        "query": "What is shown on {answer_1}'s title cover?",
                        "has_dependency": True,
                        "depends_on": [1],
                    },
                ],
            },
            "original question",
        )

        self.assertEqual(len(plan["sub_questions"]), 2)
        self.assertFalse(plan["sub_questions"][0]["has_dependency"])
        self.assertEqual(plan["sub_questions"][1]["depends_on"], [1])
        self.assertTrue(plan["sub_questions"][1]["has_dependency"])

    def test_unknown_dependency_is_not_injected_into_query(self):
        resolved, summary, recovered = _simple_resolve_dependency_question(
            {
                "id": 2,
                "question": "What is visible on the cover of {answer_1}?",
                "depends_on": [1],
            },
            {
                1: {
                    "answer": "I don't know.",
                    "resolved_question": "Which title is relevant?",
                    "rationale": "No evidence.",
                }
            },
            "What is the person on the relevant cover doing?",
        )

        self.assertTrue(recovered)
        self.assertNotIn("cover of I don't know", resolved)
        self.assertIn("Focused sub-question", resolved)
        self.assertIn("I don't know.", summary)

    async def test_adaptive_corrector_rewrites_and_retries_single_hop(self):
        async def planner_model_func(**_kwargs):
            return json.dumps(
                {
                    "rationale": "The query is single-hop but underspecified.",
                    "is_single_hop": True,
                    "reconstructed_query": "Which television title credits S. Cole as Muriel?",
                    "sub_queries": [],
                }
            )

        retry_state = SubQuestionState(
            id=1,
            original_question="What title did S. Cole appear in?",
            resolved_question="Which television title credits S. Cole as Muriel?",
            chosen_strategy="text_only",
            confidence=0.9,
        )
        retry_raw = {
            "sub_answer": {"answer": "The Bill", "rationale": "Credit evidence.", "confidence": 0.9},
            "strategy_decision": {"strategy": "text_only"},
            "text_answer": "The Bill",
            "image_answer": "",
        }
        failed = {
            "id": 1,
            "question": "What title did S. Cole appear in?",
            "resolved_question": "What title did S. Cole appear in?",
            "answer": "I don't know.",
            "failure_type": "unknown_subanswer",
            "chosen_strategy": "text_only",
            "text_evidence": [],
            "image_evidence": [],
        }

        with patch(
            "src.pipeline.query_executor._simple_execute_one_attempt",
            new=AsyncMock(return_value=(retry_state, retry_raw)),
        ) as execute:
            corrected, events = await _simple_correct_failed_subquestion(
                original_query="What is the person on the relevant cover doing?",
                node={"id": 1, "question": "What title did S. Cole appear in?", "depends_on": []},
                failed_result=failed,
                previous_summary="",
                stores={},
                funcs={},
                config=SimpleNamespace(),
                execution_context={},
                planner_model_func=planner_model_func,
                depth=0,
                max_depth=1,
            )

        self.assertEqual(corrected["answer"], "The Bill")
        self.assertEqual(corrected["correction_type"], "rewrite_and_retry")
        self.assertEqual(execute.await_count, 1)
        self.assertTrue(any(event["event"] == "adaptive_correction_executed" for event in events))

    def test_only_simple_method_is_supported(self):
        self.assertEqual(normalize_method("cor_simple"), "cor_simple")
        self.assertEqual(normalize_method("simple_cor"), "cor_simple")
        for method in ("cor_candidate", "candidate_cor", "corprorag", "bridgerag", "SGN"):
            with self.assertRaises(ValueError):
                normalize_method(method)

    async def test_cor_simple_runs_final_original_question_retrieval(self):
        async def planner_model_func(**_kwargs):
            return json.dumps(
                {
                    "reasoning_plan": "Resolve the title, then inspect its cover.",
                    "sub_queries": [
                        {"query": "Which title is relevant?", "has_dependency": False, "depends_on": []}
                    ],
                }
            )

        async def llm_model_func(**_kwargs):
            return json.dumps({"answer": "writing", "rationale": "The final image evidence shows writing."})

        final_state = SubQuestionState(
            id=900000,
            original_question="What is the person doing?",
            resolved_question="What is the person doing?",
            chosen_strategy="image_only",
            confidence=0.9,
        )
        final_raw = {
            "sub_answer": {"answer": "writing", "rationale": "Visible in the image.", "confidence": 0.9},
            "strategy_decision": {"strategy": "image_only"},
            "text_answer": "",
            "image_answer": "writing",
        }
        sub_results = [
            {
                "id": 1,
                "question": "Which title is relevant?",
                "resolved_question": "Which title is relevant?",
                "answer": "Example Title",
                "rationale": "Text evidence.",
            }
        ]
        config = SimpleNamespace(
            cor_simple_max_correction_depth=1,
            simple_planner_model_name="gpt-4o-mini",
            llm_model_name="Qwen3-VL",
        )
        with patch(
            "src.pipeline.query_executor._run_simple_natural_leg",
            new=AsyncMock(return_value=(sub_results, [])),
        ), patch(
            "src.pipeline.query_executor._simple_execute_one_attempt",
            new=AsyncMock(return_value=(final_state, final_raw)),
        ) as execute:
            result = await cor_simple_query(
                query="What is the person doing?",
                knowledge_graph_inst=None,
                entities_vdb=None,
                relationships_vdb=None,
                chunks_vdb=None,
                text_chunks_db=None,
                global_config={
                    "llm_model_func": llm_model_func,
                    "planner_model_func": planner_model_func,
                    "vlm_model_func": None,
                    "rerank_texts_func": None,
                    "config": config,
                },
            )

        self.assertEqual(execute.await_count, 1)
        self.assertFalse(execute.await_args.kwargs["allow_direct_answer"])
        self.assertEqual(result["answer"], "writing")
        self.assertEqual(result["method_variant"], "cor_simple_text_aligned_multimodal")
        self.assertFalse(result["logic_harness"]["enabled"])
        self.assertEqual(result["final_sgn_retrieval"]["answer"], "writing")


if __name__ == "__main__":
    unittest.main()
