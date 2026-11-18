import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from src.pipeline import query_executor as q
from src.retrieval.fusion import _extract_keywords_relationships
from src.llm import openai_complete_if_cache

class EdgeFixTests(unittest.IsolatedAsyncioTestCase):
    def test_among_selector_and_factual_arrays(self):
        for question in ['Who among ["Alice", "Bob"] wears glasses?',
                         'Who is standing next to a trophy among ["Alice", "Bob"]?']:
            self.assertTrue(q._simple_question_is_visual_selector(question))
            self.assertEqual(q._simple_extract_candidate_set_from_question(question), ['Alice','Bob'])
        self.assertEqual(q._simple_bound_image_targets('What role in {answer_1}?',
                         'What role in ["Hello, Dolly!", "Film B"]?'), ['Hello, Dolly!', 'Film B'])
        for value in ['[]', '[1, 2]', '["unfinished']:
            self.assertEqual(q._simple_bound_image_targets('What role in {answer_1}?','What role in '+value+'?'), [])

    async def test_evidence_aliases_propagate_to_image_matching(self):
        context={}
        text={'evidence':[{'content':'Row 6: 2009 | Inspector Lewis (Lewis (TV series)) | Dorian Crane\nThe race was held in Murcia, Spain.'}]}
        with patch.object(q,'_run_text_sgn_once',AsyncMock(return_value=text)):
            await q._simple_text_retrieve('question',1,{}, {},SimpleNamespace(),context)
        for title, alias in [('Lewis (TV series)','Inspector Lewis'),('Lewis (TV series)','Inspector Lewis (Lewis (TV series))'),('Murcia','Murcia, Spain')]:
            result={'candidate_state':{'candidates':[{'page_title':title,'image_id':'id'}]}}
            with patch.object(q,'_run_image_retrieve',AsyncMock(return_value=result)):
                retrieved=await q._simple_image_retrieve('question',2,{}, {},SimpleNamespace(),context)
            item=q._compact_image_evidence(retrieved)[0]
            self.assertTrue(q._simple_image_matches_candidate_name(alias,item))
            self.assertFalse(q._simple_image_matches_candidate_name(alias,{'page_title':'Unrelated film'}))
        self.assertFalse(q._simple_image_matches_candidate_name('Murcia, Spain',{'page_title':'Murcia'}))
        self.assertFalse(q._simple_image_matches_candidate_name('Ali',{'page_title':'Aliens'}))

    async def test_invalid_json_retry_bypasses_both_bad_caches(self):
        calls=[]
        async def model(**kwargs):
            calls.append(kwargs)
            return '{"keywords":' if len(calls)<4 else '{"keywords":["valid"]}'
        result=await _extract_keywords_relationships('query',model)
        self.assertEqual(result,{'keywords':['valid']})
        self.assertEqual([c.get('bypass_cache',False) for c in calls],[False,False,True,True])

    async def test_cache_bypass_is_local_option_and_refreshes_cache(self):
        cache=SimpleNamespace(get_by_id=AsyncMock(return_value={'return':'bad'}),upsert=AsyncMock())
        create=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='fresh'))]))
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        args=dict(model='test',prompt='test',hashing_kv=cache,openai_client=client)
        self.assertEqual(await openai_complete_if_cache(**args),'bad')
        create.assert_not_awaited()
        self.assertEqual(await openai_complete_if_cache(**args,bypass_cache=True),'fresh')
        self.assertNotIn('bypass_cache',create.call_args.kwargs)
        self.assertEqual(cache.get_by_id.await_count,1)
        cache.upsert.assert_awaited_once()

if __name__=='__main__':unittest.main()
