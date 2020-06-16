import argparse
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def classwise_evaluate(
    dataset_path: str,
    prediction_path: str) -> dict:
    """Classwise COCO evaluation.
    """
    dataset = COCO(dataset_path)
    prediction = dataset.loadRes(prediction_path)
    prediction_path = Path(prediction_path)
    task = prediction_path.name.split('.')[0]
    assert task in ('bbox', 'segm'), f'Invalid task {task}.'
    evaluator = COCOeval(dataset, prediction, task)
    results = dict()
    for category_id, category_info in dataset.cats.items():
        category_name = category_info['name']
        evaluator.params.catIds = [category_id]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics = [f'{v}_{category_name}' for v in ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl')]
        results.update({k: v for k, v in zip(metrics, evaluator.stats)})
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate per-class APs.")
    parser.add_argument(
        "prediction",
        metavar="P",
        help="path to prediction json file",
    )
    parser.add_argument(
        "--dataset",
        default="datasets/coco/annotations/instances_val2017.json",
        metavar="D",
        help="path to validation json file",
    )

    args = parser.parse_args()
    per_class_APs = classwise_evaluate(args.dataset, args.prediction)
    voc_category_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
        'bus', 'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
        'bottle', 'chair', 'couch', 'potted plant', 'dining table', 'tv']
    all_set = dict()
    voc_set = dict()
    non_voc_set = dict()
    for k, v in per_class_APs.items():
        metric, name = k.split('_')
        if not metric in all_set.keys():
            all_set[metric] = []
        all_set[metric].append(v)
        if not metric in voc_set.keys():
            voc_set[metric] = []
        if name in voc_category_names:
            voc_set[metric].append(v)
        if not metric in non_voc_set.keys():
            non_voc_set[metric] = []
        if not name in voc_category_names:
            non_voc_set[metric].append(v)

    print('\n')
    for set_name, select_set in [
        ('All set (80 categories)', all_set),
        ('VOC set (20 categories)', voc_set),
        ('Non-VOC set (60 categories)', non_voc_set)]:
        print(set_name)
        print(', '.join([f'{metric}: {100 * sum(select_set[metric]) / len(select_set[metric]):.1f}' \
            for metric in ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']]))
