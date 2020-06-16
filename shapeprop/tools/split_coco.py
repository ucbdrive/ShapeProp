import json
import argparse
from copy import deepcopy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split COCO.")
    parser.add_argument(
        "--coco-root",
        default="datasets/coco",
        metavar="ROOT",
        help="root path of COCO dataset",
    )
    args = parser.parse_args()

    print('Spliting COCO17 dataset into voc/non-voc versions for the class-wise semi-supervision setting...')
    # Load COCO17 dataset
    with open(f'{args.coco_root}/annotations/instances_train2017.json', 'r') as f:
        dataset = json.load(f)
    # Category name to id mapping
    category_name_to_id = {v['name']: v['id'] for v in dataset['categories']}
    # Define 20 categories overlapped with PASCAL VOC
    voc_category_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
        'bus', 'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
        'bottle', 'chair', 'couch', 'potted plant', 'dining table', 'tv']
    assert len(voc_category_names) == 20
    voc_category_ids = set([category_name_to_id[v] for v in voc_category_names])
    # Get 60 non-voc categories
    non_voc_category_names = [v['name'] for v in dataset['categories'] if not v['name'] in voc_category_names]
    assert len(non_voc_category_names) == 60
    non_voc_category_ids = set([category_name_to_id[v] for v in non_voc_category_names])
    # Split coco annotations into voc/non-voc versions
    voc_annotations = deepcopy(dataset['annotations'])
    for anno in voc_annotations:
        if not anno['category_id'] in voc_category_ids:
            # remove masks
            del anno['segmentation']
    non_voc_annotations = deepcopy(dataset['annotations'])
    for anno in non_voc_annotations:
        if not anno['category_id'] in non_voc_category_ids:
            # remove masks
            del anno['segmentation']
    # Save to files
    with open(f'{args.coco_root}/annotations/instances_train2017_voc_category.json', 'w') as f:
        dataset['annotations'] = voc_annotations
        json.dump(dataset, f)
    with open(f'{args.coco_root}/annotations/instances_train2017_non_voc_category.json', 'w') as f:
        dataset['annotations'] = non_voc_annotations
        json.dump(dataset, f)
