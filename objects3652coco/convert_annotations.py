import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

import utils


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert Objects365 annotations into MS Coco format"
    )
    parser.add_argument("-p", "--path", dest="path", help="path to objects365 data", type=str)
    parser.add_argument(
        "--apply-exif",
        dest="apply_exif",
        action="store_true",
        help="apply the exif orientation correctly",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["val", "train"],
        choices=["train", "val", "test"],
        help="subsets to convert",
    )
    parser.add_argument(
        "--task", type=str, default="bbox", choices=["bbox", "panoptic"], help="type of annotations"
    )
    args = parser.parse_args()
    return args


args = parse_args()
base_dir = args.path

if args.apply_exif:
    print("-" * 60)
    print("We will apply exif orientation...")
    print("-" * 60)

if not isinstance(args.subsets, list):
    args.subsets = [args.subsets]


for subset in args.subsets:
    # Convert annotations
    print("converting {} data".format(subset))

    # Select correct source files for each subset
    if subset == "train":
        annotation_sourcefile = "zhiyuan_objv2_train.json"

    elif subset == "val":
        annotation_sourcefile = "zhiyuan_objv2_val.json"

    # Load original annotations
    print("loading original annotations ...", end="\r")
    annotation_sourcefile = os.path.join(base_dir, "annotations", annotation_sourcefile)
    original_annotations = json.load(open(annotation_sourcefile, "r"))
    print("loading original annotations ... Done")

    print(original_annotations.keys())
    oi = {}

    # Add basic dataset info
    print("adding basic dataset info")

    # Add license information
    print("adding basic license info")
    oi["licenses"] = original_annotations["licenses"]

    # Convert category information
    print("converting category info")
    oi["categories"] = original_annotations["categories"]

    # Convert image mnetadata
    print("converting image info ...")
    image_dir = os.path.join(base_dir, subset)
    images = []
    removed_image_ids = []

    # path_images = []
    # for root, dirs, files in os.walk(image_dir):
    #     for name in files:
    #         if name.endswith('jpg'):
    #             path_images.append(os.path.join(root, name))
    #             if len(path_images) % 100000 == 0:
    #                 print("total images: ", len(path_images))
    # print("total images: ", len(path_images))

    # for img in tqdm(original_annotations['images']):
    #     file_name = os.path.join(image_dir, img['file_name'])
    #     if file_name in path_images:
    #         images.append(img)
    #     else:
    #         removed_image_ids.append(img['id'])
    #         print("\nremoving: ", file_name, img)

    bad_images = [
        "images/v2/patch16/objects365_v2_00908726.jpg",
        "images/v1/patch6/objects365_v1_00320532.jpg",
        "images/v1/patch6/objects365_v1_00320534.jpg",
        "images/v1/patch15/objects365_v1_00712985.jpg",
        "images/v2/patch45/objects365_v2_02081009.jpg",
        "images/v2/patch16/objects365_v2_00904061.jpg",
        "images/v1/patch3/objects365_v1_00162953.jpg",
        "images/v2/patch26/objects365_v2_01339848.jpg",
        "images/v2/patch31/objects365_v2_01524152.jpg",
        "images/v1/patch3/objects365_v1_00175464.jpg",
        "images/v2/patch44/objects365_v2_02037230.jpg",
    ]
    images = original_annotations["images"]
    for i in tqdm(range(len(images) - 1, -1, -1)):
        if images[i]["file_name"] in bad_images:
            img = images.pop(i)
            removed_image_ids.append(img["id"])
            print("\nremoving: ", img)
        else:
            pass

    if args.apply_exif:
        for img in tqdm(images):
            file_name = os.path.join(image_dir, img["file_name"])
            image = utils.read_image(file_name, format="BGR")
            if image.shape[1] != img["width"] or image.shape[0] != img["height"]:
                print("before exif correction: ", img)
                img["width"], img["height"] = image.shape[1], image.shape[0]
                print("after exif correction: ", img)

    oi["images"] = images

    # Convert instance annotations
    print("converting annotations ...")
    # Convert annotations
    if args.task == "bbox":
        annotations = original_annotations["annotations"]
        for i in tqdm(range(len(annotations) - 1, -1, -1)):
            if annotations[i]["image_id"] in removed_image_ids:
                ann = annotations.pop(i)
                print("\nremoving: ", ann)
            else:
                pass
        oi["annotations"] = annotations
    elif args.task == "panoptic":
        assert 0

    # Write annotations into .json file
    filename = os.path.join(base_dir, "annotations/", "objects365_{}.json".format(subset))
    print("writing output to {}".format(filename))
    json.dump(oi, open(filename, "w"))
    print("Done")
