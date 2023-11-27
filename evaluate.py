import json

from language_evaluation import CocoEvaluator


if __name__ == '__main__':
    with open("data/captions_val2017.json", "r") as f:
        ann = json.load(f)

    with open("output/result.json", "r") as f:
        result = json.load(f)

    gold = []
    pred = []

    for cap in ann['annotations']:
        gold.append(cap['caption'])
        pred.append(result[str(cap['image_id'])])
        print(gold[-1])
        print(pred[-1])

    # evaluator = CocoEvaluator()

    # score = evaluator.run_evaluation(pred, gold)

    # with open('output/score.json', 'w') as f:
    #     json.dump(score, f)
    

