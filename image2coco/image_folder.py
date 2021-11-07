import argparse
import json
import os
import os.path as osp
import pathlib
import random
from distutils.util import strtobool

import numpy as np
from PIL import Image

import utils

folder_classes = []


def cvt_annotations(path_h_w, out_file, has_instance, has_segmentation):
    label_ids = {name: i for i, name in enumerate(folder_classes)}
    print("label_ids: ", label_ids)

    annotations = []

    file_client_args = dict(backend="disk")
    color_type = "color"

    for i, [img_path, height, width] in enumerate(path_h_w):
        if i % 1000 == 0:
            print(i)
        # if i > 10000:
        #     break
        # print(i, img_path)

        wnid = pathlib.PurePath(img_path).parent.name

        if has_instance:
            bboxes = [[1, 1, width, height]]
            labels = [label_ids[wnid]]

            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        else:
            bboxes = (np.zeros((0, 4), dtype=np.float32),)
            labels = (np.zeros((0,), dtype=np.int64),)

        annotation = {
            "filename": img_path,
            "width": width,
            "height": height,
            "ann": {
                "bboxes": bboxes.astype(np.float32),
                "labels": labels.astype(np.int64),
                "bboxes_ignore": np.zeros((0, 4), dtype=np.float32),
                "labels_ignore": np.zeros((0,), dtype=np.int64),
            },
        }

        annotations.append(annotation)
    annotations = cvt_to_coco_json(annotations, has_segmentation)

    with open(out_file, "w") as f:
        json.dump(annotations, f)

    return annotations


def cvt_to_coco_json(annotations, has_segmentation):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco["images"] = []
    coco["type"] = "instance"
    coco["categories"] = []
    coco["annotations"] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        if has_segmentation:
            annotation_item["segmentation"] = []

            seg = []
            # bbox[] is x1,y1,x2,y2
            # left_top
            seg.append(int(bbox[0]))
            seg.append(int(bbox[1]))
            # left_bottom
            seg.append(int(bbox[0]))
            seg.append(int(bbox[3]))
            # right_bottom
            seg.append(int(bbox[2]))
            seg.append(int(bbox[3]))
            # right_top
            seg.append(int(bbox[2]))
            seg.append(int(bbox[1]))

            annotation_item["segmentation"].append(seg)

        xywh = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item["area"] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item["ignore"] = 0
            annotation_item["iscrowd"] = 1
        else:
            annotation_item["ignore"] = 0
            annotation_item["iscrowd"] = 0
        annotation_item["image_id"] = int(image_id)
        annotation_item["bbox"] = xywh.astype(int).tolist()
        annotation_item["category_id"] = int(category_id)
        annotation_item["id"] = int(annotation_id)
        coco["annotations"].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(folder_classes):
        category_item = dict()
        category_item["supercategory"] = str("none")
        category_item["id"] = int(category_id)
        category_item["name"] = str(name)
        coco["categories"].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict["filename"]
        ann = ann_dict["ann"]
        assert file_name not in image_set
        image_item = dict()
        image_item["id"] = int(image_id)
        image_item["file_name"] = str(file_name)
        image_item["height"] = int(ann_dict["height"])
        image_item["width"] = int(ann_dict["width"])
        coco["images"].append(image_item)
        image_set.add(file_name)

        bboxes = ann["bboxes"][:, :4]
        labels = ann["labels"]
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann["bboxes_ignore"][:, :4]
        labels_ignore = ann["labels_ignore"]
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco


def get_filename_key(x):

    basename = os.path.basename(x)
    if basename[:-4].isdigit():
        return int(basename[:-4])

    return basename


def _strtobool(x):
    return bool(strtobool(x))


def parse_args():
    parser = argparse.ArgumentParser(description="Convert image list to coco format")
    parser.add_argument("--img-root", help="img root", required=True)
    parser.add_argument("--out-file", help="output path", required=True)
    parser.add_argument("--key-path", help="key path")
    parser.add_argument("--filter-file", help="filter file")
    parser.add_argument("--category-file", help="reset categories by this json file")
    parser.add_argument('--max-per-dir', nargs='?', const=1e10, type=int, default=1e10, help="max per dir")
    parser.add_argument('--num-labels', nargs='?', const=0, type=int, default=0, help="number of labels")
    parser.add_argument('--has-instance', nargs='?', const=True, type=_strtobool, default=True, help="has instance")
    parser.add_argument('--has-segmentation', nargs='?', const=True, type=_strtobool, default=True, help="has segmentation")
    parser.add_argument('--has-shuffle', nargs='?', const=False, type=_strtobool, default=False, help="shuffle or sort")
    args = parser.parse_args()
    return args


def main():
    global folder_classes

    args = parse_args()
    print(args)

    img_root = args.img_root
    out_file = args.out_file

    key_path = args.key_path
    category_file = args.category_file
    filter_file = args.filter_file

    max_per_dir = args.max_per_dir
    num_labels = args.num_labels

    has_instance = args.has_instance
    has_segmentation = args.has_segmentation

    has_shuffle = args.has_shuffle

    if filter_file:
        with open(filter_file, 'r') as f:
            keeps = [line.strip() for line in f.readlines()]
    else:
        keeps = None

    path_h_w = []
    cnt = 0
    for root, dirs, files in os.walk(img_root):
        files = [f for f in files if f.endswith(('.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG'))]

        if has_shuffle:
            random.shuffle(files)
        else:
            files = sorted(files, key=lambda x: get_filename_key(x), reverse=False)

        cnt_per_dir = 0
        for name in files:
            # if cnt >= 10:
            #     break

            if max_per_dir > 0 and cnt_per_dir >= max_per_dir:
                break

            path = os.path.join(root, name)
            rpath = path.replace(img_root, "")

            if keeps and rpath not in keeps:
                continue

            if key_path and key_path not in path:
                continue

            wnid = pathlib.PurePath(path).parent.name

            if wnid not in folder_classes:
                folder_classes.append(wnid)
            try:
                img = utils.read_image(path, format="BGR")
                height, width, _ = img.shape
            except Exception as e:
                print("*" * 60)
                print("fail to open image: ", e)
                print("*" * 60)
                continue

            if width < 224 or height < 224:
                continue

            # if width >  1000 or height > 1000:
            #     continue

            path_h_w.append([rpath, height, width])

            # print(rpath, os.path.basename(rpath))
            cnt_per_dir += 1
            cnt += 1

            if cnt % 1000 == 0:
                print(cnt)

        print("folder: ", root, " number: ", cnt_per_dir)

        # for dirname in dirs:
        #     folder_classes.append(dirname)

    folder_classes.sort()
    for i in range(len(folder_classes), num_labels):
        folder_classes.append("unknown_" + str(i))

    print("categories", len(folder_classes), ":", folder_classes)
    print("image number: ", cnt)

    annotations = cvt_annotations(path_h_w, out_file, has_instance, has_segmentation)
    print("Done!")

    if category_file:
        print("Reset categories!")

        print("loading", category_file)
        with open(category_file, "r") as f:
            json_data = json.load(f)
            categories = json_data["categories"]

        all_id = [x["id"] for x in categories]
        min_id = min(all_id)
        for category in categories:
            category["id"] -= min_id

        annotations["categories"] = categories

        with open(out_file, "w") as f:
            json.dump(annotations, f)

    print(args)


if __name__ == "__main__":
    main()
