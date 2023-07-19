import os
import clip

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy
from wilds.datasets.wilds_dataset import WILDSDataset
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from imagenet_class_names import imagenet_classes

IMAGENET_CLASS_DICT = {
    'baby pacifier': 680,
    'can opener': 473}

CONFUSED_CLASS_DICT = {
    'baby pacifier': 898,
    'can opener': 623}

METADATA_MAP_DICT = {
    'baby pacifier': {
            'generic-spurious': ['bottle', 'baby'], # Padding for str formatting
            'spurious': ['baby'],
            'y': ['water bottle', 'baby pacifier']
        },
    'can opener': {
            'generic-spurious': ['opener', 'can'], # Padding for str formatting
            'spurious': ['can'],
            'y': ['letter opener', 'can opener']
        },
}

# METADATA_MAP_DICT['can opener']['y'].extend(imagenet_classes)


def detect_imagenet_attr(dataset, attr, score_threshold=0.1):
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", load_in_8bit=True)
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").cuda()
    texts = [f"a photo of a {a}" for a in attr]

    detect = np.zeros(len(dataset))
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            image = sample['image']
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

            if np.sum(results[0]["scores"].detach().cpu().numpy() >= score_threshold) > 0:
                detect[i] = 1
    
    print(f'Total detected: {np.sum(detect)}/{len(detect)}')
    del processor
    del model
    return detect


class ImagenetSpurious(WILDSDataset):
    _dataset_name = 'imagenetspurious'
    _versions_dict = {
        '1.0': {}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official', imagenet_class='baby pacifier', seed=42, bingeval=False, commercial=False):
        self._version = version
        os.makedirs(root_dir, exist_ok=True)
        self._data_dir = root_dir

        # Get the y values
        self._y_size = 1
        self._class_indices = np.array([CONFUSED_CLASS_DICT[imagenet_class], IMAGENET_CLASS_DICT[imagenet_class]])

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        self._metadata_fields = ['generic-spurious', 'y']
        self._metadata_map = METADATA_MAP_DICT[imagenet_class]

        class_name = ''.join(imagenet_class.split())
        os.makedirs(os.path.join(self.data_dir, 'imagenet'), exist_ok=True)

        self.bingeval = bingeval
        if bingeval:
            preload_dataset_path = os.path.join(self.data_dir, f'imagenet/preload_dataset_{class_name}_bing.pth')
        else:
            preload_dataset_path = os.path.join(self.data_dir, f'imagenet/preload_dataset_{class_name}.pth')
        if not os.path.exists(preload_dataset_path):
            datasets_ = []
            y = []
            spurious = []
            splits = []
            for split in ['train', 'validation']:
                dataset = load_dataset("imagenet-1k", use_auth_token=True, cache_dir=self._data_dir, split=split)
                for i, class_idx in enumerate(self._class_indices):
                    print(f"Processing class {class_idx} {split}...")
                    dataset_ = dataset.filter(lambda x: x["label"]==class_idx)
                    datasets_.append(dataset_)
                    y.append(np.ones(len(dataset_)) * i)
                    detect = detect_imagenet_attr(dataset_, self._metadata_map['spurious'], score_threshold=0.1)
                    spurious.append(detect)
                    if split == 'train':                        
                        train_split = np.zeros(len(dataset_))
                        if not bingeval:
                            for j in [0, 1]:
                                np.random.seed(seed)
                                val_idx = np.random.choice(np.where(detect==j)[0], 25, replace=False)
                                train_split[val_idx] = 1
                        splits.append(train_split)
                    else:
                        if bingeval:
                            splits.append(np.ones(len(dataset_)) * self.DEFAULT_SPLITS['val'])
                        else:
                            splits.append(np.ones(len(dataset_)) * self.DEFAULT_SPLITS['test'])
            self._input_dataset = concatenate_datasets(datasets_)
            print(f"Saving the dataset to {preload_dataset_path}")
            torch.save(self._input_dataset, preload_dataset_path)

            metadata_df = pd.DataFrame({'y': np.concatenate(y), 'spurious': np.concatenate(spurious), 'split': np.concatenate(splits)})
            metadata_df.to_csv(os.path.join(self.data_dir, f'imagenet/preload_dataset_{class_name}.csv'), index=False)
            
        else:
            print(f"Loading the class dataset from {preload_dataset_path}")
            self._input_dataset = torch.load(preload_dataset_path)
            metadata_df= pd.read_csv(os.path.join(self.data_dir, f'imagenet/preload_dataset_{class_name}.csv'))

        if bingeval:
            spurious_str = self._metadata_map['spurious'][0]
            if commercial:
                self.bingeval_dataset = ImageNetBing(os.path.join(root_dir, f'imagenet/{spurious_str}-commercial'), transform=None) 
            else:
                self.bingeval_dataset = ImageNetBing(os.path.join(root_dir, f'imagenet/{spurious_str}'), transform=None) 
            self.bingeval_start = len(self._input_dataset)
            metadata_df = pd.concat([
                metadata_df,
                pd.DataFrame({'y': self.bingeval_dataset._metadata_df['y'].values, 'spurious': self.bingeval_dataset._metadata_df['spurious'].values, 'split': np.ones(len(self.bingeval_dataset)) * self.DEFAULT_SPLITS['test']})], ignore_index=True)

        self._y_array = torch.LongTensor(metadata_df['y'].values).flatten()
        self._n_classes = len(self._class_indices)

        self._metadata_array = torch.stack(
            (torch.LongTensor(metadata_df['spurious']), self._y_array),
            dim=1
        )

        # Extract filenames
        self._original_resolution = (224, 224)

        self._split_array = metadata_df['split'].values

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['generic-spurious', 'y']))

        super().__init__(root_dir, download, split_scheme)


    def __getitem__(self, index):
        if self.bingeval and index >= (self.bingeval_start):
            x, y, metadata = self.bingeval_dataset[index-self.bingeval_start]
        else:
            sample = self._input_dataset.__getitem__(int(index))
            x = sample['image']
            y = self.y_array[index]
            metadata = self.metadata_array[index]

        return x, y, metadata

    def get_input(self, idx):
        if self.bingeval and idx >= (self.bingeval_start):
            x, _, _ = self.bingeval_dataset[idx-self.bingeval_start]
        else:
            sample = self._input_dataset.__getitem__(int(idx))
            x = sample['image']

        return x

    def get_original_filename(self, idx):
        sample = self._input_dataset._getitem(int(idx), decoded=False)
        path = sample['image']['path']
        path = '_'.join(path.split('_')[:2])
        return path

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)

        results, results_str = self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

        acc = 0.
        total = 0.
        for yi, y in enumerate(self._metadata_map['y'][:2]):
            y = y.replace('=', ':').replace(',','_').replace(' ','')
            for si, s in enumerate(self._metadata_map['generic-spurious']):
                s = s.replace('=', ':').replace(',','_').replace(' ','')
                num = np.sum(self._split_array[np.where((self._metadata_array[:, 0]==si) & (self._metadata_array[:, 1]==yi))[0]]==0)
                acc += results[f'acc_y:{y}_generic-spurious:{s}'] * num
                total += num 
        results['adj_acc_avg'] = acc / total

        results_str = f"Adjusted average acc: {results['adj_acc_avg']:.3f}\n" + '\n'.join(results_str.split('\n')[1:])

        return results, results_str


class ImageNet(Dataset):
    def __init__(self, dataset, transform):
        self.imagenet = dataset

        self.labels = np.array(self.imagenet["label"])
        self.image_indices = np.arange(len(dataset))
        self.transform = transform

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, index):
        image_index = self.image_indices[index]
        sample = self.imagenet.__getitem__(int(image_index))
        image = self.transform(sample['image'])
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] > 3:
            image = image[:3, :, :]

        label_index = self.labels[image_index]
        
        return image, label_index


class ImageNetBing(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root=root, transform=transform)
        self._metadata_df = pd.read_csv(os.path.join(root, 'metadata.csv'))
        self._metadata_array = torch.stack(
            (torch.LongTensor(self._metadata_df['spurious'].values).flatten(), torch.LongTensor(self._metadata_df['y'].values).flatten()),
            dim=1
        )
        self.transform = transform

        from_source_domain = torch.as_tensor(
            [0 for _ in range(len(self._metadata_array))],
            dtype=torch.int64
        ).unsqueeze(dim=1)
        self._metadata_array = torch.cat(
            [self._metadata_array, from_source_domain],
            dim=1
        )

    def __getitem__(self, index):
        path = os.path.join(self.root, self._metadata_df['filename'].values[index])
        target = self._metadata_df['y'].values[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        metadata = self._metadata_array[index]

        return sample, target, metadata


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


def evaluate_imagenet(algorithm, loader):
    try:
        zeroshot_weights = zeroshot_classifier(algorithm.model.module.model, imagenet_classes, algorithm.model.module.templates)
    except:
        zeroshot_weights = zeroshot_classifier(algorithm.model.model, imagenet_classes, algorithm.model.templates)
    correct = 0.
    count = 0.

    with torch.no_grad():

        for _, (images, target) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()
            
            # predict
            try:
                image_features = algorithm.model.model.encode_image(images)
            except:
                image_features = algorithm.model.module.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            preds = torch.argmax(logits, dim=-1)
            acc = preds.eq(target).float().detach().cpu().numpy()

            correct += np.sum(acc)
            count += len(target)

    return correct / count
    
    