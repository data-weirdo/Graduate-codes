import os
import sys
from datasets import load_metric, load_dataset

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.join(dirname, 'data')
sys.path.append(dirname)

from process import postprocess_qa_predictions


def calculate_score(datasets, valid_data, raw_predictions, wandb):
    final_predictions = postprocess_qa_predictions(datasets["validation"], valid_data, raw_predictions)
    metric = load_metric("squad")
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
    results = metric.compute(predictions=formatted_predictions, references=references)
    print(results)

    wandb.log({"EM": results})

    return results