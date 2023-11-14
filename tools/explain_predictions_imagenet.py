import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, nargs='+', default=[0])
parser.add_argument('--model', choices=['RN50'], default='RN50')
parser.add_argument('--split', choices=['train', 'validation'], default='validation')
parser.add_argument('--class_rank_range', type=int, nargs='+', default=[0, 0])
parser.add_argument('--data_dir', default='./data')
parser.add_argument('--save_dir', default='./data/imagenet_mi')

parser.add_argument('--evaluate', action='store_true', help='Run evaluation')

# Detection
parser.add_argument('--object_only', action='store_true')
parser.add_argument('--score_threshold', type=float, default=0.1)
parser.add_argument('--min_detect', type=int, default=10)

# GradCAM 
parser.add_argument('--gradcam', action='store_true', help='Run GradCAM.')
parser.add_argument('--gradcam_save', action='store_true', help='Save GradCAM images.')
parser.add_argument('--gradcam_save_dir', type=str, default='./data/imagenet_mi/gradcam/', help='GradCAM save path')
parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
parser.add_argument('--eigen_smooth', action='store_true', help='Reduce noise by taking the first principle componenet of cam_weights*activations')
parser.add_argument('--method', type=str, default='gradcam')

# Summary
parser.add_argument('--compute_mi', action='store_true')
parser.add_argument('--min_acc_diff', type=float, default=0.1)
parser.add_argument('--log_attr_per_class', type=int, default=5)

config = parser.parse_args()

import os
device_str = ",".join(map(str, config.device))
os.environ["CUDA_VISIBLE_DEVICES"] = device_str

print('Devices: ', device_str)

import clip
import cv2
import datasets
datasets.logging.set_verbosity(datasets.logging.ERROR)
from datasets import load_dataset
import json
import numpy as np
import pandas as pd
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from imagenet_helpers import imagenet_classes, imagenet_templates, ImageNetSubset, compute_mi
from gradcam import GradCAM, show_cam_on_image

# predict
# heatmap, save heatmap image and mask
# detect, for each detected object, compare box and heatmap mask

from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode
img_transform = Compose([
        Resize(size=224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size=(224, 224))])

def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def evaluate_imagenet_class(dataset, class_idx, model, preprocess, zeroshot_weights, batch_size=128, num_workers=4, gradcam=False):
    print(f"Evaluating class {class_idx}...")
    class_images = ImageNetSubset(dataset, class_idx, preprocess)
    loader = torch.utils.data.DataLoader(class_images, batch_size=batch_size, num_workers=num_workers)
    correct = np.zeros(len(class_images))
    confused = np.zeros(len(class_images))

    with torch.no_grad():

        for _, (images, target, indices) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()
            
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            preds = torch.argmax(logits, dim=-1)
            acc = preds.eq(target).float().detach().cpu().numpy()

            correct[indices] = acc
            confused[indices] = torch.argsort(logits, dim=-1, descending=True)[:, 1].float().detach().cpu().numpy()

    return correct, confused

def np_vec_no_jit_iou(boxes1, boxes2):
    def run(bboxes1, bboxes2):
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
        return iou

def main(config):
    class_indices = np.arange(config.class_rank_range[0], config.class_rank_range[1])
    clip_model, preprocess = clip.load(config.model)
    dataset = load_dataset("imagenet-1k", use_auth_token=True, cache_dir=config.data_dir, split=config.split)

    # prepare zero-shot weight
    df = pd.read_csv(os.path.join(config.data_dir, f'imagenet_{config.model.lower()}.csv'))
    weight_path = os.path.join(config.data_dir, f'clip_zeroshot_weights_{config.model}_{config.split}.pt')
    if not os.path.exists(weight_path):
        print(f"Calculating zero-shot weights...")
        zeroshot_weights = zeroshot_classifier(clip_model, imagenet_classes, imagenet_templates)
        torch.save(zeroshot_weights, weight_path)
    else:
        zeroshot_weights = torch.load(weight_path)

    # evaluate by class
    for class_idx in df.sort_values(by=['top-5', 'top-1'])['class index'].values[class_indices]:
        if config.evaluate:        
            correct, confused = evaluate_imagenet_class(dataset, class_idx, clip_model, preprocess, zeroshot_weights, gradcam=config.gradcam)
            print(f"Class {class_idx}: {imagenet_classes[class_idx]}")
            np.save(os.path.join(config.save_dir, f'correct_{config.split}_class{class_idx}_{config.model}.npy'), correct)
            np.save(os.path.join(config.save_dir, f'confused_{config.split}_class{class_idx}_{config.model}.npy'), confused)

    vocab = []
    attributes = json.load(open(os.path.join(config.data_dir, 'attribute_synsets.json'), 'r'))
    vocab.extend(set([' '.join(attributes[a].split('.')[0].split('_')) for a in attributes]))
    objects = json.load(open(os.path.join(config.data_dir, 'object_synsets.json'), 'r'))
    vocab.extend(set([' '.join(objects[a].split('.')[0].split('_')) for a in objects]))
    vocab = list(set(vocab))
    vocab.sort()
    print('Vocabulary size: ', len(vocab))

    texts = [f"a photo of a {t}" for t in vocab]

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", load_in_8bit=True)
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").cuda()

    for i, class_idx in enumerate(df.sort_values(by=['top-5', 'top-1'])['class index'].values[class_indices]):
        print(f"Detecting class {class_idx}...")
        detect = detect_imagenet_class(config, dataset, processor, model, clip_model, preprocess, zeroshot_weights, texts, vocab, class_idx=class_idx, score_threshold=config.score_threshold)
        np.save(os.path.join(config.save_dir, f'detect_{config.split}_class{class_idx}_{config.model}_threshold{config.score_threshold}.npy'), detect)

    # generate a summary
    if config.compute_mi:
        mi_df = compute_mi(config, vocab, objects, np.arange(1000))
        if config.object_only:
            mi_df.to_csv(os.path.join(config.save_dir, f'mi_{config.split}_{config.model}_{config.score_threshold}_object.csv'))
        else:
            mi_df.to_csv(os.path.join(config.save_dir, f'mi_{config.split}_{config.model}_{config.score_threshold}.csv'))


def detect_imagenet_class(config, dataset, processor, model, clip_model, preprocess, zeroshot_weights, texts, vocab, class_idx=516, score_threshold=0.1):
    class_dataset = dataset.filter(lambda x: x["label"] == class_idx)
    detect = np.zeros((len(class_dataset), len(vocab)))

    if config.gradcam and config.gradcam_save:
        class_str = imagenet_classes[class_idx].replace(' ', '_')
        save_dir = os.path.join(config.gradcam_save_dir, f'{class_idx}_{class_str}')
        os.makedirs(save_dir, exist_ok=True)
        torch.autograd.set_detect_anomaly(True)

    for i, sample in enumerate(class_dataset):
        image = sample['image']
        image = img_transform(image)

        with torch.no_grad():
            try:
                inputs = processor(text=texts, images=image, return_tensors="pt")
            except:
                continue
            for key in inputs:
                inputs[key] = inputs[key].cuda()
        
            outputs = model(**inputs)

            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([image.size[::-1]])
            # Convert outputs (bounding boxes and class logits) to COCO API
            results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

            valid = np.where(results[0]["scores"].detach().cpu().numpy() >= score_threshold)[0]
            objs = results[0]["labels"].detach().cpu().numpy()[valid]
            
            detect[i, objs] = 1

        if config.gradcam:
            targets = None
            # Using the with statement ensures the context is freed, and you can
            # recreate different CAM objects in a loop.
            with GradCAM(model=clip_model,
                            target_layers=[clip_model.visual.layer4],
                            use_cuda=True) as cam:

                grayscale_cam = cam(input_tensor=preprocess(image).unsqueeze(0),
                                    zeroshot_weights=zeroshot_weights,
                                    targets=targets,  
                                    aug_smooth=config.aug_smooth,
                                    eigen_smooth=config.eigen_smooth)

                # Here grayscale_cam has only one image in the batch
                grayscale_cam = grayscale_cam[0, :]

                rgb_img = np.array(image, dtype=np.uint8)[:, :, ::-1]
                cam_image = show_cam_on_image(np.array(image, dtype=np.float32)/255, grayscale_cam.astype(np.float32), use_rgb=True)

                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            if config.gradcam_save:
                cv2.imwrite(f'{save_dir}/{i}_ori.jpg', rgb_img)
                cv2.imwrite(f'{save_dir}/{i}_cam.jpg', cam_image)

            boxes = results[0]["boxes"].detach().cpu().numpy()[valid]
            if len(boxes) > 0:
                # compare detected bounding boxes with the cam mask                
                best_score = 0.
                best_box_mask = np.zeros_like(grayscale_cam)
                
                for box_i, box in enumerate(boxes):
                    box[box < 0] = 0
                    box[box > 224] = 224
                    box = box.astype(int)
                    box_mask = np.zeros_like(grayscale_cam)
                    box_mask[box[1]:box[3], box[0]:box[2]] = 1
                    score = np.sum(box_mask * grayscale_cam) / (np.sum(box_mask) + np.sum(grayscale_cam))
                    if score > best_score:
                        best_box_mask = box_mask
                        best_score = score
                        best_box_predict = objs[box_i]

                detect[i, best_box_predict] += 1
                cam_image = show_cam_on_image(np.array(image, dtype=np.float32)/255, best_box_mask.astype(np.float32), use_rgb=True)

                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                vocab_str = vocab[best_box_predict].replace(' ', '_')
                cv2.imwrite(f'{save_dir}/{i}_box_{vocab_str}.jpg', cam_image)

    return detect


if __name__ == '__main__':
    main(config)
    print('Finish!')