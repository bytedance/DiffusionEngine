import os.path as osp
import json
from tqdm import tqdm

from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata


DE_DATASETS = {
    'coco-de': ('COCO-DE', None),
}


def post_process_DE_dataset(root, ann_src=None, ann_tar=None, tar_class_names=None, force=False):
    images_dir = osp.join(root, 'images')
    if not ann_src:
        ann_src = osp.join(root, 'annotations_pre_process.json')
    if not ann_tar:
        ann_tar = osp.join(root, 'annotations.json')

    if not force and osp.exists(ann_tar):
        # print('Target Ann File Exists.', ann_tar)
        # print('Use force=True to rewrite')
        return images_dir, ann_tar

    with open(ann_src) as f:
        # "ann_id" and "categories" haven't been set yet
        annotations = json.load(f)

    meta = _get_builtin_metadata('coco')
    things = meta["thing_classes"]
    reverse_id_map = {v: k for k, v in meta["thing_dataset_id_to_contiguous_id"].items()}
    if tar_class_names is not None:
         # coco_contiguous_to_tar_things
         reverse_id_map = {things.index(c): i for i, c in enumerate(tar_class_names)}

    # only preserve annotations whose cid is in tar_class
    valid_ids = reverse_id_map.keys()

    # set ann_id, re-map category_id, filter extra anns
    new_annotations = []
    image_id_with_anns = set()
    for _, ann in tqdm(enumerate(annotations['annotations']), total=len(annotations['annotations'])):
        if ann['category_id'] in valid_ids:
            ann['id'] = len(new_annotations)
            ann['category_id'] = reverse_id_map[ann['category_id']]
            new_annotations.append(ann)
            image_id_with_anns.add(ann['image_id'])
    annotations['annotations'] = new_annotations

    # set categories， get meta data from ref. dataset
    annotations['categories'] = [{'id': reverse_id_map[i], 'name': t} for i, t in enumerate(things) if i in valid_ids]

    new_images = []
    for _, img in tqdm(enumerate(annotations['images']), total=len(annotations['images'])):
        if img['id'] in image_id_with_anns:
            new_images.append(img)
    print(f'Filter {len(annotations["images"]) - len(new_images)} Images without valid Ann')
    annotations['images'] = new_images

    with open(ann_tar, 'w') as f:
        f.write(json.dumps(annotations))
        f.flush()

    return images_dir, ann_tar


def register_all_coco_de(root):
    for key, (parent_dir, fn_json) in DE_DATASETS.items():
        if fn_json and osp.exists(osp.join(root, parent_dir, fn_json)):
            image_root = osp.join(root, parent_dir, 'images')
            json_file = osp.join(root, parent_dir, fn_json)
        else:
            image_root, json_file = post_process_DE_dataset(osp.join(root, parent_dir))
        register_coco_instances(
            key, _get_builtin_metadata('coco'), json_file, image_root
        )


register_all_coco_de(root='engine_output')
