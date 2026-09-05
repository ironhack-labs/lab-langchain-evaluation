import unittest
from copy import deepcopy
from evaluation_lab import *

class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs=load_clothing();cls.cases=clothing_examples(cls.docs);cls.qa=make_qa(Retriever(cls.docs))
    def test_data_and_gold(self):
        self.assertEqual(len(self.docs),1000)
        self.assertEqual(len(self.cases),4)
        self.assertEqual(len(nyc_examples(load_nyc())),5)
    def test_retrieval(self):
        result=self.qa.invoke(self.cases[0]['query'])
        self.assertIn('product-10',[d.metadata['id'] for d in result['source_documents']])
        self.assertEqual(result['mode'],'local')
    def test_metrics(self):
        self.assertEqual(token_f1('same answer','same answer'),1)
        self.assertEqual(token_f1('wrong','correct'),0)
        prediction=self.qa.invoke(self.cases[0]['query'])
        corrupt=deepcopy(prediction);corrupt['source_documents']=[]
        self.assertEqual(local_metrics(self.cases[0],corrupt)['retrieval_hit_at_k'],0)
    def test_no_reference_leak(self):
        a=self.qa.invoke({'query':self.cases[0]['query'],'answer':'INJECTED REFERENCE'})
        b=self.qa.invoke({'query':self.cases[0]['query']})
        self.assertEqual(a['result'],b['result'])
        self.assertNotIn('INJECTED REFERENCE',a['result'])
    def test_ragas_schema(self):
        from ragas import SingleTurnSample
        from ragas.metrics import Faithfulness,ResponseRelevancy,LLMContextRecall,LLMContextPrecisionWithReference
        p=self.qa.invoke(self.cases[0]['query'])
        row=ragas_rows(self.cases[:1],[p])[0]
        sample=SingleTurnSample(**row)
        self.assertTrue(all(isinstance(x,str) for x in sample.retrieved_contexts))
        self.assertEqual(sample.reference,'Yes')
if __name__=='__main__':unittest.main()
