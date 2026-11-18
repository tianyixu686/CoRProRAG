import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.pipeline.query_executor import (
    _simple_execute_one_attempt, _simple_image_retrieve, _simple_text_retrieve, cor_simple_query,
)
from src.pipeline.simple_state import SubQuestionState


class CoreAblationTest(unittest.IsolatedAsyncioTestCase):
    def stores(self):
        return {
            "knowledge_graph_inst": AsyncMock(), "entities_vdb": AsyncMock(),
            "relationships_vdb": AsyncMock(), "chunks_vdb": AsyncMock(),
            "text_chunks_db": AsyncMock(), "images_vdb": AsyncMock(),
            "image_text_vdb": AsyncMock(), "image_chunks_db": AsyncMock(),
            "visual_entities_db": AsyncMock(),
        }

    async def test_without_sgn_uses_only_plain_vector_and_kv_stores(self):
        stores = self.stores()
        stores["chunks_vdb"].query.return_value = [{"id": "c1"}]
        stores["text_chunks_db"].get_by_id.return_value = {"content": "Evidence text."}
        stores["images_vdb"].query.return_value = [{"id": "i1", "__metrics__": 0.8}]
        stores["image_chunks_db"].get_by_id.return_value = {
            "image_path": "/tmp/i.jpg", "caption": "Image from page 'Example'."}
        config = SimpleNamespace(cor_simple_ablation_variant="without_sgn", kg_top_k=8,
                                 image_retrieval_candidate_k=8, mm_sgn_context_max_tokens=1000)
        funcs = {"llm_model_func": AsyncMock(return_value='{"answer":"Example"}')}
        with patch("src.pipeline.query_executor._run_text_sgn_once", new=AsyncMock()) as text_sgn, \
             patch("src.pipeline.query_executor._run_image_retrieve", new=AsyncMock()) as image_sgn:
            text = await _simple_text_retrieve("Question", 1, stores, funcs, config, {})
            image = await _simple_image_retrieve("Question", 1, stores, funcs, config, {})
        self.assertEqual(text["retrieval_backend"], "plain_dense_text")
        self.assertEqual(image["candidate_state"]["candidates"][0]["page_title"], "Example")
        text_sgn.assert_not_awaited()
        image_sgn.assert_not_awaited()
        stores["knowledge_graph_inst"].get_node.assert_not_awaited()
        stores["entities_vdb"].query.assert_not_awaited()
        stores["relationships_vdb"].query.assert_not_awaited()

    async def test_without_sgn_directly_reads_top8_text_and_top5_raw_images(self):
        stores = self.stores()
        stores["chunks_vdb"].query.return_value = [{"id": "c1"}]
        stores["text_chunks_db"].get_by_id.return_value = {"content": "Evidence text."}
        stores["images_vdb"].query.return_value = [
            {"id": f"i{i}", "__metrics__": 1 - i / 10} for i in range(5)]
        stores["image_chunks_db"].get_by_id.side_effect = [
            {"image_path": f"/tmp/i{i}.jpg", "caption": f"Image from page 'Page{i}'.",
             "global_description": "precomputed KG description"} for i in range(5)]
        reader = AsyncMock(return_value='{"answer":"Page1","confidence":0.8}')
        vlm = AsyncMock(return_value="Visible object")
        config = SimpleNamespace(cor_simple_ablation_variant="without_sgn", mm_sgn_context_max_tokens=1000)
        funcs = {"llm_model_func": reader, "vlm_model_func": vlm}
        with patch("src.pipeline.query_executor._simple_decide_strategy", new=AsyncMock()) as strategy:
            state, raw = await _simple_execute_one_attempt(
                original_query="Question", node_id=1, original_node_question="Question",
                resolved_question="Question", previous_summary="", stores=stores,
                funcs=funcs, config=config, execution_context={})
        stores["chunks_vdb"].query.assert_awaited_once_with("Question", top_k=8)
        stores["images_vdb"].query.assert_awaited_once_with("Question", top_k=5)
        self.assertEqual(vlm.await_count, 5)
        self.assertEqual(len(state.image_evidence), 5)
        self.assertTrue(all(not item["global_description"] for item in state.image_evidence))
        self.assertEqual(raw["sub_answer"]["answer"], "Page1")
        strategy.assert_not_awaited()
        stores["knowledge_graph_inst"].get_node.assert_not_awaited()

    async def test_sgn_only_skips_planner_and_retrieves_original_once(self):
        stores = self.stores()
        planner = AsyncMock(side_effect=AssertionError("planner must not run"))
        reader = AsyncMock(return_value=json.dumps({"answer": "Example"}))
        state = SubQuestionState(id=900000, original_question="Question", resolved_question="Question",
                                 chosen_strategy="text_only")
        run_once = AsyncMock(return_value=(state, {"sub_answer": {"answer": "Example"}}))
        config = SimpleNamespace(cor_simple_ablation_variant="sgn_only",
                                 cor_simple_final_output_tokens=256, cor_simple_final_input_tokens=6200)
        with patch("src.pipeline.query_executor._simple_execute_one_attempt", run_once):
            result = await cor_simple_query("Question", stores["knowledge_graph_inst"], stores["entities_vdb"],
                stores["relationships_vdb"], stores["chunks_vdb"], stores["text_chunks_db"],
                {"config": config, "llm_model_func": reader, "planner_model_func": planner},
                images_vdb=stores["images_vdb"], image_chunks_db=stores["image_chunks_db"])
        self.assertEqual(run_once.await_count, 1)
        planner.assert_not_awaited()
        self.assertEqual(result["ablation_variant"], "sgn_only")
        self.assertEqual(result["simple_plan"]["sub_questions"], [])

    async def test_without_ac_passes_zero_depth_without_other_pipeline_change(self):
        stores = self.stores()
        planner = AsyncMock(return_value='{"sub_queries":[{"question":"Question"}]}')
        reader = AsyncMock(return_value='{"answer":"Example"}')
        state = SubQuestionState(id=900000, original_question="Question", resolved_question="Question",
                                 chosen_strategy="text_only")
        run_once = AsyncMock(return_value=(state, {"sub_answer": {"answer": "Example"}}))
        run_leg = AsyncMock(return_value=([], []))
        config = SimpleNamespace(cor_simple_ablation_variant="without_ac", cor_simple_max_correction_depth=1,
                                 cor_simple_final_output_tokens=256, cor_simple_final_input_tokens=6200)
        with patch("src.pipeline.query_executor._simple_execute_one_attempt", run_once), \
             patch("src.pipeline.query_executor._run_simple_natural_leg", run_leg):
            result = await cor_simple_query("Question", stores["knowledge_graph_inst"], stores["entities_vdb"],
                stores["relationships_vdb"], stores["chunks_vdb"], stores["text_chunks_db"],
                {"config": config, "llm_model_func": reader, "planner_model_func": planner})
        self.assertEqual(run_leg.await_args.kwargs["max_depth"], 0)
        self.assertFalse(result["adaptive_corrector"]["enabled"])

    async def test_saved_full_plan_replays_without_new_planner_call(self):
        stores = self.stores()
        planner = AsyncMock(side_effect=AssertionError("replay must not call planner"))
        reader = AsyncMock(return_value='{"answer":"Example"}')
        state = SubQuestionState(id=900000, original_question="Question", resolved_question="Question",
                                 chosen_strategy="text_only")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "full.json"
            path.write_text(json.dumps([{"query": "Question", "answer": {"simple_plan": {
                "reasoning_plan": "Saved", "sub_questions": [{"id": 1, "question": "Which example?", "depends_on": []}]
            }}}]))
            config = SimpleNamespace(cor_simple_ablation_variant="without_sgn", cor_simple_plan_replay_path=str(path),
                                     cor_simple_max_correction_depth=1,
                                     cor_simple_final_output_tokens=256, cor_simple_final_input_tokens=6200)
            with patch("src.pipeline.query_executor._simple_execute_one_attempt",
                       AsyncMock(return_value=(state, {"sub_answer": {"answer": "Example"}}))), \
                 patch("src.pipeline.query_executor._run_simple_natural_leg", AsyncMock(return_value=([], []))) as run_leg:
                result = await cor_simple_query("Question", stores["knowledge_graph_inst"], stores["entities_vdb"],
                    stores["relationships_vdb"], stores["chunks_vdb"], stores["text_chunks_db"],
                    {"config": config, "llm_model_func": reader, "planner_model_func": planner})
        planner.assert_not_awaited()
        self.assertEqual(run_leg.await_args.kwargs["plan"]["sub_questions"][0]["question"], "Which example?")
        self.assertEqual(result["plan_source"], "replay")


if __name__ == "__main__":
    unittest.main()
