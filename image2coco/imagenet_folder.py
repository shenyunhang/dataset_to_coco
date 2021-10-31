import argparse
import json
import os
import os.path as osp
import pathlib

import numpy as np
from nltk.corpus import wordnet as wn
from PIL import Image

import utils

# import nltk
# nltk.download('wordnet')


folder_classes = []


def cvt_annotations(img_name_paths, out_file, has_labels):
    label_ids = {name: i for i, name in enumerate(folder_classes)}
    print("label_ids: ", label_ids)

    annotations = []

    file_client_args = dict(backend="disk")
    color_type = "color"

    for i, [img_name, img_path, height, width] in enumerate(img_name_paths):
        if i % 1000 == 0:
            print(i)
        # if i > 10000:
        #     break
        # print(i, img_path)

        wnid = pathlib.PurePath(img_path).parent.name

        if has_labels:
            bboxes = [[1, 1, width, height]]
            labels = [label_ids[wnid]]

            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        else:
            bboxes = (np.zeros((0, 4), dtype=np.float32),)
            labels = (np.zeros((0,), dtype=np.int64),)

        annotation = {
            "filename": os.path.join(wnid, img_name),
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
    annotations = cvt_to_coco_json(annotations)

    with open(out_file, "w") as f:
        json.dump(annotations, f)

    return annotations


def cvt_to_coco_json(annotations):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Convert image list to coco format")
    parser.add_argument("--img-root", help="img root")
    parser.add_argument("--out-file", help="output path")
    parser.add_argument("--max-per-cat", help="max per cat")
    parser.add_argument("--num-labels", help="number of labels")
    parser.add_argument(
        "--category-file",
        help="reset categories according to this category json file path by matching wordnet id",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    out_file = args.out_file
    img_root = args.img_root
    max_per_cat = int(args.max_per_cat)
    has_labels = True
    num_labels = int(args.num_labels)

    category_file = args.category_file

    global folder_classes

    fns = []
    cnt = 0
    wnid = ""
    for root, dirs, files in os.walk(img_root):
        cnt_per_dir = 0
        for name in files:
            if cnt % 1000 == 0:
                print(cnt)

            # if cnt >= 10:
            #     break

            if max_per_cat > 0 and cnt_per_dir >= max_per_cat:
                break

            if name.endswith(".jpg") or name.endswith(".JPG"):
                pass
            elif name.endswith(".png") or name.endswith(".PNG"):
                pass
            elif name.endswith(".jpeg") or name.endswith(".JPEG"):
                pass
            else:
                print("skipping: ", name)
                continue

            path = os.path.join(root, name)

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

            # if width < 224 or height < 224:
            #     continue

            # if width >  1000 or height > 1000:
            #     continue

            fns.append([name, path, height, width])

            # print(name, path, pathlib.PurePath(path).parent.name)
            cnt_per_dir += 1
            cnt += 1

        print("folder: ", wnid, " number: ", cnt_per_dir)

        # for dirname in dirs:
        #     folder_classes.append(dirname)

    folder_classes.sort()
    for i in range(len(folder_classes), num_labels):
        folder_classes.append("unknown_" + str(i))

    print("categories", len(folder_classes), ":", folder_classes)
    print("image number: ", cnt)

    annotations = cvt_annotations(fns, out_file, has_labels)
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

        wnid_to_new_category_id = {}
        for category in categories:
            name = category["name"]
            synset = category["synset"]

            category_id = category["id"]

            try:
                synset = wn.synset(synset)

                offset = synset.offset()
                wnid = "n{:08d}".format(offset)

                category["wnid"] = wnid
                wnid_to_new_category_id[wnid] = category_id

            except Exception as e:
                category["wnid"] = name
                wnid_to_new_category_id[name] = category_id
                print(e)

        old_category_id_to_wnid = {}
        for category in annotations["categories"]:
            old_category_id_to_wnid[category["id"]] = category["name"]

        for anno in annotations["annotations"]:
            wnid = old_category_id_to_wnid[anno["category_id"]]

            category_id = wnid_to_new_category_id[wnid]

            anno["category_id"] = category_id

        annotations["categories"] = categories

        with open(out_file, "w") as f:
            json.dump(annotations, f)


if __name__ == "__main__":
    main()
