import nest_asyncio
import os
import openai
from dotenv import load_dotenv
import pandas as pd
from rag import retrieval_chain

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
nest_asyncio.apply()

class DataLoader:
    def load_data(self, doc):
        questions = doc['Questions'].tolist()
        answers = doc['Answers'].tolist()

        examples = [
            {"query": q, "ground_truths": [answers[i]]}
            for i, q in enumerate(questions)
        ]

        return examples

class ChainEvaluator:
    def evaluate_chain(self, examples, chain):
        predictions = chain.batch(examples)
        return predictions

class MetricsEvaluator:
    def __init__(self, faithfulness_metric, answer_relevancy_metric, context_precision_metric, context_recall_metric):
        self.faithfulness_metric = faithfulness_metric
        self.answer_relevancy_metric = answer_relevancy_metric
        self.context_precision_metric = context_precision_metric
        self.context_recall_metric = context_recall_metric

    def evaluate_metrics(self, examples, predictions):
        faithfulness_chain = RagasEvaluatorChain(metric=self.faithfulness_metric)
        answer_rel_chain = RagasEvaluatorChain(metric=self.answer_relevancy_metric)
        context_rel_chain = RagasEvaluatorChain(metric=self.context_precision_metric)
        context_recall_chain = RagasEvaluatorChain(metric=self.context_recall_metric)

        faithfulness_score = faithfulness_chain.evaluate(examples, predictions)
        answer_relevancy_score = answer_rel_chain.evaluate(examples, predictions)
        context_precision_score = context_rel_chain.evaluate(examples, predictions)
        context_recall_score = context_recall_chain.evaluate(examples, predictions)

        return faithfulness_score, answer_relevancy_score, context_precision_score, context_recall_score


class DataFrameCreator:
    def create_dataframe(self, qna, faithfulness_scores, answer_relevancy_scores, context_precision_scores, context_recall_scores):
        df = pd.DataFrame({
            "Faithfulness Score": faithfulness_scores,
            "Answer Relevancy Score": answer_relevancy_scores,
            "Context Precision Score": context_precision_scores,
            "Context Recall Score": context_recall_scores
        })

        result_df = pd.concat([qna, df], axis=1)
        return result_df

class Main:
    def __init__(self):
        self.data_loader = DataLoader()
        self.chain_evaluator = ChainEvaluator()
        self.metrics_evaluator = MetricsEvaluator()
        self.data_frame_creator = DataFrameCreator()

    def run(self):
        chain = configure_retrieval_chain()

        qna = pd.read_csv('your_qna_file.csv')
        examples = self.data_loader.load_data(qna)

        predictions = self.chain_evaluator.evaluate_chain(examples, chain)

        faithfulness_score, answer_relevancy_score, context_precision_score, context_recall_score = self.metrics_evaluator.evaluate_metrics(examples, predictions)

        result_df = self.data_frame_creator.create_dataframe(qna, faithfulness_score, answer_relevancy_score, context_precision_score, context_recall_score)
        return result_df

if __name__ == "__main__":
    result_dataframe = Main().run()
    print(result_dataframe)
