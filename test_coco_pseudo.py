import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap, get_phrases_from_posmap_2

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

from pycocotools.coco import COCO
from pycocotools import mask



def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }



def getImgId(name, load_dict):
	# load_dict = json.load(open(path, 'r'))
	images = load_dict['images']

	for i in range(len(images)):
		file_name = images[i]['file_name'].split('.')[0]
		if file_name == name:
				#print(images[i])
				return images[i]['id']

def get_coco_gt(name):
    img_id = getImgId(name, coco.dataset)
    cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

    prompt_list = []

    for instance in cocotarget:
        cat = instance['category_id']
        prompt_list.append(id_to_name_dict[cat])
    return prompt_list


classes = [{"supercategory": "person", "id": 1, "name": "person"}, # 一共80类
               {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
               {"supercategory": "vehicle", "id": 3, "name": "car"},
               {"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
               {"supercategory": "vehicle", "id": 5, "name": "airplane"},
               {"supercategory": "vehicle", "id": 6, "name": "bus"},
               {"supercategory": "vehicle", "id": 7, "name": "train"},
               {"supercategory": "vehicle", "id": 8, "name": "truck"},
               {"supercategory": "vehicle", "id": 9, "name": "boat"},
               {"supercategory": "outdoor", "id": 10, "name": "traffic light"},
               {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"},
               {"supercategory": "outdoor", "id": 13, "name": "stop sign"},
               {"supercategory": "outdoor", "id": 14, "name": "parking meter"},
               {"supercategory": "outdoor", "id": 15, "name": "bench"},
               {"supercategory": "animal", "id": 16, "name": "bird"},
               {"supercategory": "animal", "id": 17, "name": "cat"},
               {"supercategory": "animal", "id": 18, "name": "dog"},
               {"supercategory": "animal", "id": 19, "name": "horse"},
               {"supercategory": "animal", "id": 20, "name": "sheep"},
               {"supercategory": "animal", "id": 21, "name": "cow"},
               {"supercategory": "animal", "id": 22, "name": "elephant"},
               {"supercategory": "animal", "id": 23, "name": "bear"},
               {"supercategory": "animal", "id": 24, "name": "zebra"},
               {"supercategory": "animal", "id": 25, "name": "giraffe"},
               {"supercategory": "accessory", "id": 27, "name": "backpack"},
               {"supercategory": "accessory", "id": 28, "name": "umbrella"},
               {"supercategory": "accessory", "id": 31, "name": "handbag"},
               {"supercategory": "accessory", "id": 32, "name": "tie"},
               {"supercategory": "accessory", "id": 33, "name": "suitcase"},
               {"supercategory": "sports", "id": 34, "name": "frisbee"},
               {"supercategory": "sports", "id": 35, "name": "skis"},
               {"supercategory": "sports", "id": 36, "name": "snowboard"},
               {"supercategory": "sports", "id": 37, "name": "sports ball"},
               {"supercategory": "sports", "id": 38, "name": "kite"},
               {"supercategory": "sports", "id": 39, "name": "baseball bat"},
               {"supercategory": "sports", "id": 40, "name": "baseball glove"},
               {"supercategory": "sports", "id": 41, "name": "skateboard"},
               {"supercategory": "sports", "id": 42, "name": "surfboard"},
               {"supercategory": "sports", "id": 43, "name": "tennis racket"},
               {"supercategory": "kitchen", "id": 44, "name": "bottle"},
               {"supercategory": "kitchen", "id": 46, "name": "wine glass"},
               {"supercategory": "kitchen", "id": 47, "name": "cup"},
               {"supercategory": "kitchen", "id": 48, "name": "fork"},
               {"supercategory": "kitchen", "id": 49, "name": "knife"},
               {"supercategory": "kitchen", "id": 50, "name": "spoon"},
               {"supercategory": "kitchen", "id": 51, "name": "bowl"},
               {"supercategory": "food", "id": 52, "name": "banana"},
               {"supercategory": "food", "id": 53, "name": "apple"},
               {"supercategory": "food", "id": 54, "name": "sandwich"},
               {"supercategory": "food", "id": 55, "name": "orange"},
               {"supercategory": "food", "id": 56, "name": "broccoli"},
               {"supercategory": "food", "id": 57, "name": "carrot"},
               {"supercategory": "food", "id": 58, "name": "hot dog"},
               {"supercategory": "food", "id": 59, "name": "pizza"},
               {"supercategory": "food", "id": 60, "name": "donut"},
               {"supercategory": "food", "id": 61, "name": "cake"},
               {"supercategory": "furniture", "id": 62, "name": "chair"},
               {"supercategory": "furniture", "id": 63, "name": "couch"},
               {"supercategory": "furniture", "id": 64, "name": "potted plant"},
               {"supercategory": "furniture", "id": 65, "name": "bed"},
               {"supercategory": "furniture", "id": 67, "name": "dining table"},
               {"supercategory": "furniture", "id": 70, "name": "toilet"},
               {"supercategory": "electronic", "id": 72, "name": "tv"},
               {"supercategory": "electronic", "id": 73, "name": "laptop"},
               {"supercategory": "electronic", "id": 74, "name": "mouse"},
               {"supercategory": "electronic", "id": 75, "name": "remote"},
               {"supercategory": "electronic", "id": 76, "name": "keyboard"},
               {"supercategory": "electronic", "id": 77, "name": "cell phone"},
               {"supercategory": "appliance", "id": 78, "name": "microwave"},
               {"supercategory": "appliance", "id": 79, "name": "oven"},
               {"supercategory": "appliance", "id": 80, "name": "toaster"},
               {"supercategory": "appliance", "id": 81, "name": "sink"},
               {"supercategory": "appliance", "id": 82, "name": "refrigerator"},
               {"supercategory": "indoor", "id": 84, "name": "book"},
               {"supercategory": "indoor", "id": 85, "name": "clock"},
               {"supercategory": "indoor", "id": 86, "name": "vase"},
               {"supercategory": "indoor", "id": 87, "name": "scissors"},
               {"supercategory": "indoor", "id": 88, "name": "teddy bear"},
               {"supercategory": "indoor", "id": 89, "name": "hair drier"},
               {"supercategory": "indoor", "id": 90, "name": "toothbrush"}]

id_to_name_dict = {}
id_to_cls_dict = {}
name_to_cls_dict = {}
for index, item in enumerate(classes):
    category_id = item['id']
    category_name = item['name']
    id_to_name_dict[category_id] = category_name
    id_to_cls_dict[category_id] = index + 1
    name_to_cls_dict[category_name] = index + 1 

# add some propmt here:
# class_dict 


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    
    # breakpoint()
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):

        pred_phrase = get_phrases_from_posmap_2(logit > text_threshold, logit, tokenized, tokenlizer)

        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def save_mask_data(output_dir, mask_list, box_list, label_list, name, H, W):
    value = 0  # 0 for background

    mask_img = torch.zeros(H, W)
    for idx, mask in enumerate(mask_list):
        class_name, logit = label_list[idx].split('(')
       
        if class_name in name_to_cls_dict:
            class_idx = name_to_cls_dict[class_name]
            mask_img[mask.cpu().numpy()[0] == True] = class_idx 
    
    out = mask_img.numpy().astype(np.uint8)

    pseudo = Image.fromarray(out)
    out_name = os.path.join(output_dir, '{}.png'.format(name))
    pseudo.save(out_name)
    
    return out



    # plt.figure(figsize=(10, 10))
    # plt.imshow(mask_img.numpy())
    # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, '{}_mask.jpg'.format(name)), bbox_inches="tight", dpi=300, pad_inches=0.0)
    # plt.close()

    # json_data = [{
    #     'value': value,
    #     'label': 'background'
    # }]

    # for label, box in zip(label_list, box_list):
    #     value += 1
    #     name, logit = label.split('(')
    #     logit = logit[:-1] # the last is ')'
    #     json_data.append({
    #         'value': value,
    #         'label': name,
    #         'logit': float(logit),
    #         'box': box.numpy().tolist(),
    #     })
    # with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
    #     json.dump(json_data, f)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--img_list", type=str, default='metadata/coco/train_04.txt')
    parser.add_argument(
        "--output_dir", "-o", type=str, default="coco2",  help="output directory"
    )

    args = parser.parse_args()

    base_dir = '/home/notebook/data/personal/S9050086/coco/'
    ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format('train', 2014))
    # coco = COCO(ann_file)
    # coco_mask = mask
    
    # img_list = os.listdir('/home/notebook/data/personal/S9050086/coco/train2014/')

    with open(args.img_list) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    print(len(img_list))
    # cls_labels_dict = np.load('metadata/cls_labels.npy',allow_pickle=True).item()
    
    # cfg
    output_dir = args.output_dir

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    preds, gts = [], []
    for index, i in enumerate(img_list):

        name = i

        if os.path.isfile(os.path.join(output_dir, "{}.png".format(name))):
            print(index, name)
            pseudo = np.asarray(Image.open(os.path.join(output_dir, "{}.png".format(name))))
        else:
            continue
            # print()
            
        gt = np.asarray(Image.open(os.path.join('/home/notebook/data/personal/S9050086/coco/coco_seg_anno/', '{}.png'.format(name[15:]))), dtype=np.int32)

        preds += list(pseudo)
        gts += list(gt)
    
    score = scores(gts, preds, n_class=81)
    with open('coco_pseudo_score.txt', "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)
    


       


