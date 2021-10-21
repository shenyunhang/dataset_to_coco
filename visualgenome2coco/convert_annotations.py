import argparse
import json
import math
import os
import pprint
import string
from collections import Counter
from math import floor

import numpy as np
from tqdm import tqdm

import utils


def preprocess_object_labels(data, alias_dict={}):
    for img in data:
        for obj in img["objects"]:
            obj["ids"] = [obj["object_id"]]
            names = []
            for name in obj["names"]:
                label = sentence_preprocess(name)
                if label in alias_dict:
                    label = alias_dict[label]
                names.append(label)
            obj["names"] = names


def extract_object_token(data, num_tokens, obj_list=[], verbose=True):
    """Builds a set that contains the object names. Filters infrequent tokens."""
    token_counter = Counter()
    for img in data:
        for region in img["objects"]:
            for name in region["names"]:
                if not obj_list or name in obj_list:
                    token_counter.update([name])
    tokens = set()
    # pick top N tokens
    token_counter_return = {}
    for token, count in token_counter.most_common():
        tokens.add(token)
        token_counter_return[token] = count
        if len(tokens) == num_tokens:
            break
    if verbose:
        print(("Keeping %d / %d objects" % (len(tokens), len(token_counter))))
    return tokens, token_counter_return


def merge_duplicate_boxes(data):
    def IoU(b1, b2):
        if b1[2] <= b2[0] or b1[3] <= b2[1] or b1[0] >= b2[2] or b1[1] >= b2[3]:
            return 0

        b1b2 = np.vstack([b1, b2])
        minc = np.min(b1b2, 0)
        maxc = np.max(b1b2, 0)
        union_area = (maxc[2] - minc[0]) * (maxc[3] - minc[1])
        int_area = (minc[2] - maxc[0]) * (minc[3] - maxc[1])
        return float(int_area) / float(union_area)

    def to_x1y1x2y2(obj):
        x1 = obj["x"]
        y1 = obj["y"]
        x2 = obj["x"] + obj["w"]
        y2 = obj["y"] + obj["h"]
        return np.array([x1, y1, x2, y2], dtype=np.int32)

    def inside(b1, b2):
        return b1[0] >= b2[0] and b1[1] >= b2[1] and b1[2] <= b2[2] and b1[3] <= b2[3]

    def overlap(obj1, obj2):
        b1 = to_x1y1x2y2(obj1)
        b2 = to_x1y1x2y2(obj2)
        iou = IoU(b1, b2)
        if all(b1 == b2) or iou > 0.9:  # consider as the same box
            return 1
        elif (inside(b1, b2) or inside(b2, b1)) and obj1["names"][0] == obj2["names"][
            0
        ]:  # same object inside the other
            return 2
        elif iou > 0.6 and obj1["names"][0] == obj2["names"][0]:  # multiple overlapping same object
            return 3
        else:
            return 0  # no overlap

    num_merged = {1: 0, 2: 0, 3: 0}
    print("merging boxes..")
    for img in data:
        # mark objects to be merged and save their ids
        objs = img["objects"]
        num_obj = len(objs)
        for i in range(num_obj):
            if "M_TYPE" in objs[i]:  # has been merged
                continue
            merged_objs = []  # circular refs, but fine
            for j in range(i + 1, num_obj):
                if "M_TYPE" in objs[j]:  # has been merged
                    continue
                overlap_type = overlap(objs[i], objs[j])
                if overlap_type > 0:
                    objs[j]["M_TYPE"] = overlap_type
                    merged_objs.append(objs[j])
            objs[i]["mobjs"] = merged_objs

        # merge boxes
        filtered_objs = []
        merged_num_obj = 0
        for obj in objs:
            if "M_TYPE" not in obj:
                ids = [obj["object_id"]]
                dims = [to_x1y1x2y2(obj)]
                prominent_type = 1
                for mo in obj["mobjs"]:
                    ids.append(mo["object_id"])
                    obj["names"].extend(mo["names"])
                    dims.append(to_x1y1x2y2(mo))
                    if mo["M_TYPE"] > prominent_type:
                        prominent_type = mo["M_TYPE"]
                merged_num_obj += len(ids)
                obj["ids"] = ids
                mdims = np.zeros(4)
                if prominent_type > 1:  # use extreme
                    mdims[:2] = np.min(np.vstack(dims)[:, :2], 0)
                    mdims[2:] = np.max(np.vstack(dims)[:, 2:], 0)
                else:  # use mean
                    mdims = np.mean(np.vstack(dims), 0)
                obj["x"] = int(mdims[0])
                obj["y"] = int(mdims[1])
                obj["w"] = int(mdims[2] - mdims[0])
                obj["h"] = int(mdims[3] - mdims[1])

                num_merged[prominent_type] += len(obj["mobjs"])

                obj["mobjs"] = None
                obj["names"] = list(set(obj["names"]))  # remove duplicates

                filtered_objs.append(obj)
            else:
                assert "mobjs" not in obj

        img["objects"] = filtered_objs
        assert merged_num_obj == num_obj

    print("# merged boxes per merging type:")
    print(num_merged)


def build_token_dict(vocab):
    """build bi-directional mapping between index and token"""
    token_to_idx, idx_to_token = {}, {}
    next_idx = 1
    vocab_sorted = sorted(list(vocab))  # make sure it's the same order everytime
    for token in vocab_sorted:
        token_to_idx[token] = next_idx
        idx_to_token[next_idx] = token
        next_idx = next_idx + 1

    return token_to_idx, idx_to_token


def sentence_preprocess(phrase):
    """preprocess a sentence: lowercase, clean up weird chars, remove punctuation"""
    replacements = {
        "½": "half",
        "—": "-",
        "™": "",
        "¢": "cent",
        "ç": "c",
        "û": "u",
        "é": "e",
        "°": " degree",
        "è": "e",
        "…": "",
    }
    # phrase = phrase.encode('utf-8')
    phrase = phrase.lstrip(" ").rstrip(" ")
    for k, v in replacements.items():
        phrase = phrase.replace(k, v)
    # return str(phrase).lower().translate(None, string.punctuation).decode('utf-8', 'ignore')
    return str(phrase).lower().translate(str.maketrans("", "", string.punctuation))


def make_alias_dict(dict_file):
    """create an alias dictionary from a file"""
    out_dict = {}
    vocab = []
    for line in open(dict_file, "r"):
        alias = line.strip("\n").strip("\r").split(",")
        alias_target = alias[0] if alias[0] not in out_dict else out_dict[alias[0]]
        for a in alias:
            out_dict[a] = alias_target  # use the first term as the aliasing target
        vocab.append(alias_target)
    return out_dict, vocab


def make_list(list_file):
    """create a blacklist list from a file"""
    return [line.strip("\n").strip("\r") for line in open(list_file)]


def filter_object_boxes(data, img_data, area_frac_thresh):
    """
    filter boxes by a box area-image area ratio threshold
    """
    thresh_count = 0
    all_count = 0
    for i, img in enumerate(data):
        filtered_obj = []
        area = float(img_data[i]["height"] * img_data[i]["width"])
        for obj in img["objects"]:
            if float(obj["h"] * obj["w"]) > area * area_frac_thresh:
                filtered_obj.append(obj)
                thresh_count += 1
            all_count += 1
        img["objects"] = filtered_obj
    print("box threshod: keeping %i/%i boxes" % (thresh_count, all_count))


def filter_by_idx(data, valid_list):
    return [data[i] for i in valid_list]


def main(args):
    print("start")
    pprint.pprint(args)

    if args.apply_exif:
        print("-" * 60)
        print("We will apply exif orientation...")
        print("-" * 60)

    base_dir = args.path
    object_alias = os.path.join(base_dir, "annotations", "object_alias.txt")
    object_input = os.path.join(base_dir, "annotations", "objects.json")
    metadata_input = os.path.join(base_dir, "annotations", "image_data.json")

    obj_alias_dict = {}
    print("using object alias from %s" % (object_alias))
    obj_alias_dict, obj_vocab_list = make_alias_dict(object_alias)

    obj_list = []
    if len(args.object_list) > 0:
        print("using object list from %s" % (args.object_list))
        obj_list = make_list(args.object_list)
        assert len(obj_list) >= args.num_objects

    # read in the annotation data
    print("loading json files..")
    print("using object data from %s" % (object_input))
    obj_data = json.load(open(object_input))
    print("using image data from %s" % (metadata_input))
    img_data = json.load(open(metadata_input))
    assert len(obj_data) == len(img_data)
    num_im = len(img_data)

    # print(obj_data[0])
    # print(img_data[0])

    # sanity check
    for i in range(num_im):
        assert obj_data[i]["image_id"] == img_data[i]["image_id"]

    print("processing %i images" % num_im)

    # preprocess label data
    preprocess_object_labels(obj_data, alias_dict=obj_alias_dict)

    if args.min_box_area_frac > 0:
        # filter out invalid small boxes
        print("threshold bounding box by %f area fraction" % args.min_box_area_frac)
        filter_object_boxes(obj_data, img_data, args.min_box_area_frac)  # filter by box dimensions

    # merge_duplicate_boxes(obj_data)

    # build vocabulary
    object_tokens, object_token_counter = extract_object_token(obj_data, args.num_objects, obj_list)

    label_to_idx, idx_to_label = build_token_dict(object_tokens)

    # print out vocabulary
    print("objects: ")
    print(object_token_counter)

    # Convert image mnetadata
    print("converting image info ...")
    images = []
    image_ids = []

    corrupted_ims = ["1592.jpg", "1722.jpg", "4616.jpg", "4617.jpg"]

    for i in tqdm(range(num_im), mininterval=0.5):
        path = img_data[i]["url"]
        path = os.path.normpath(path)
        path_split = path.split(os.sep)
        if path_split[-1] in corrupted_ims:
            print("corrupted image: ", img_data[i])
            continue

        image_id = img_data[i]["image_id"]
        coco_id = img_data[i]["coco_id"]
        flickr_id = img_data[i]["flickr_id"]

        height = img_data[i]["height"]
        width = img_data[i]["width"]

        has_obj = False
        for obj in obj_data[i]["objects"]:
            names = obj["names"]
            assert len(names) == 1
            name = names[0]
            if name not in object_tokens:
                continue
            has_obj = True
            break

        if not has_obj:
            continue

        img = {}
        img["id"] = image_id
        img["file_name"] = os.path.join(path_split[-2], path_split[-1])

        img["height"] = height
        img["width"] = width

        if args.apply_exif:
            filename = os.path.join(base_dir, img["file_name"])
            image = utils.read_image(filename, format="BGR")
            if image.shape[1] != img["width"] or image.shape[0] != img["height"]:
                print("before exif correction: ", img)
                img["width"], img["height"] = image.shape[1], image.shape[0]
                print("after exif correction: ", img)

        images.append(img)
        image_ids.append(image_id)

    # build train/val/test splits
    print("build train/val/test splits")
    num_im = len(images)
    num_im_train = int(num_im * 0.7)

    print("build train split")
    images_train = images[:num_im_train]
    image_ids_train = image_ids[:num_im_train]

    print("build val split")
    images_val = images[num_im_train:]
    image_ids_val = image_ids[num_im_train:]

    # Convert instance annotations
    print("converting annotations info ...")
    annotations = []
    annotations_train = []
    annotations_val = []
    label_to_synset = {obj: [] for obj in object_tokens}

    ann_id = 0
    for i in tqdm(range(num_im), mininterval=0.5):
        image_id = img_data[i]["image_id"]
        if image_id not in image_ids:
            continue

        for obj in obj_data[i]["objects"]:
            names = obj["names"]
            assert len(names) == 1
            name = names[0]
            if name not in object_tokens:
                continue

            print(obj)

            synsets = obj["synsets"]
            object_id = obj["object_id"]
            # merged_object_ids = obj['merged_object_ids']
            x = obj["x"]
            y = obj["y"]
            h = obj["h"]
            w = obj["w"]

            ann = {}
            ann["id"] = ann_id
            ann_id += 1

            ann["image_id"] = image_id
            ann["category_id"] = label_to_idx[name]

            # ann["bbox"] = [x, y, x + w, y + h]
            ann["bbox"] = [x, y, w, h]
            ann["area"] = h * w

            ann['iscrowd'] = False

            annotations.append(ann)
            if image_id in image_ids_train:
                annotations_train.append(ann)
            elif image_id in image_ids_val:
                annotations_val.append(ann)
            else:
                assert 0

            # assert len(synsets) <= 1
            if len(synsets) > 0:
                synset = synsets[0]
                if synset not in label_to_synset[name]:
                    label_to_synset[name].append(synset)

    print("all images: ", len(images))
    print("train images: ", len(images_train))
    print("val images: ", len(images_val))
    print("all annotations: ", len(annotations))
    print("train annotations: ", len(annotations_train))
    print("val annotations: ", len(annotations_val))

    oi_train = {}
    oi_val = {}

    # Add basic dataset info
    print("adding basic dataset info")
    oi_train["info"] = {}
    oi_val["info"] = {}

    # Add license information
    print("adding basic license info")
    oi_train["licenses"] = []
    oi_val["licenses"] = []

    # Convert category information
    print("converting category info")
    categories = []
    for i, name in idx_to_label.items():
        cat = {}
        cat["id"] = i
        cat["name"] = name
        cat["synsets"] = label_to_synset[name]
        categories.append(cat)

    oi_train["categories"] = categories
    oi_val["categories"] = categories

    # Convert image mnetadata
    print("converting image info ...")
    oi_train["images"] = images_train
    oi_val["images"] = images_val

    # Convert instance annotations
    print("converting annotations ...")
    oi_train["annotations"] = annotations_train
    oi_val["annotations"] = annotations_val

    # Write annotations into .json file
    filename = os.path.join(base_dir, "annotations/", "visualgenome_box_train.json")
    print("writing output to {}".format(filename))
    json.dump(oi_train, open(filename, "w"))

    filename = os.path.join(base_dir, "annotations/", "visualgenome_box_val.json")
    print("writing output to {}".format(filename))
    json.dump(oi_val, open(filename, "w"))
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", dest="path", help="path to objects365 data", type=str)
    parser.add_argument(
        "--apply-exif",
        dest="apply_exif",
        action="store_true",
        help="apply the exif orientation correctly",
    )
    parser.add_argument("--object_list", default="VG/object_list.txt", type=str)
    parser.add_argument(
        "--num_objects", default=150, type=int, help="set to 0 to disable filtering"
    )
    parser.add_argument("--min_box_area_frac", default=0.002, type=float)

    args = parser.parse_args()
    main(args)
