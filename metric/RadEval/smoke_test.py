from radeval import RadEval
import json

evaluator = RadEval(metrics=["radgraph", "f1chexbert", "radcliq"])
results = evaluator(
    refs=["No pleural effusions or pneumothoraces."],
    hyps=["No pleural effusions or pneumothoraces."],
)
print(json.dumps(results, indent=2))
