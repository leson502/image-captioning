import json

from language_evaluation import CocoEvaluator


if __name__ == '__main__':
    with open("output/output.json", "r") as f:
        result = json.load(f)

    evaluator = CocoEvaluator()

    score = evaluator.run_evaluation(result['pred'], result['gold'])

    with open('output/score.json', 'w') as f:
        json.dump(score, f)
