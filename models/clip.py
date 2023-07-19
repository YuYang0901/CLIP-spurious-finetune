import clip
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module

# ImageNet prompt templates used by CLIP
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

class ImageEncoder(Module):
    def __init__(self, model):
        super(ImageEncoder, self).__init__()
        self.model = model    

    def forward(self, input):
        input_features = self.model.encode_image(input)
        input_features = input_features / input_features.norm(dim=-1, keepdim=True)
        return input_features

    def encode_text(self, texts):
        embeddings = self.model.encode_text(texts) #embed with text encoder
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    def train(self, mode=True):
        self.model.train(mode)

class ZeroShotClassifier(Module):
    def __init__(self, model, metadata_map, templates=imagenet_templates):
        super(ZeroShotClassifier, self).__init__()
        self.model = model       
        self.classnames = metadata_map['y']
        self.templates = templates
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def zeroshot_classifier(self, classnames, avg=True):
        """
        Zero-shot classifier for pre-trained CLIP models.
        """
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in self.templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = self.model.encode_text(texts) #embed with text encoder
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if not avg:
                zeroshot_weights.append(class_embeddings)
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def forward(self, input_features):
        zeroshot_weights = self.zeroshot_classifier(self.classnames)
        logits = self.logit_scale.exp() * input_features @ zeroshot_weights
        return logits

    def train(self, mode=True):
        self.model.train(mode)

class ZeroShotCLIP(Module):
    def __init__(self, model, metadata_map, templates=imagenet_templates):
        super(ZeroShotCLIP, self).__init__()
        self.model = model       
        self.metadata_map = metadata_map
        self.classnames = metadata_map['y']
        self.templates = templates
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def zeroshot_classifier(self, classnames, avg=True):
        """
        Zero-shot classifier for pre-trained CLIP models.
        """
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in self.templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = self.model.encode_text(texts) #embed with text encoder
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if not avg:
                zeroshot_weights.append(class_embeddings)
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def zeroshot_weights(self):
        return self.zeroshot_classifier(self.classnames)

    def featurizer(self, input):
        input_features = self.model.encode_image(input)
        input_features = input_features / input_features.norm(dim=-1, keepdim=True)
        return input_features

    def classifier(self, input_features):
        zeroshot_weights = self.zeroshot_weights()
        logits = self.logit_scale.exp() * input_features @ zeroshot_weights
        return logits

    def forward(self, input, return_features=False):
        input_features = self.featurizer(input)
        logits = self.classifier(input_features)
        if return_features:
            return logits, input_features
        else:
            return logits
    
    def train(self, mode=True):
        self.model.train(mode)


class LinearProbCLIP(Module):
    def __init__(self, model, metadata_map, templates=imagenet_templates):
        super(LinearProbCLIP, self).__init__()
        self.model = model       
        self.metadata_map = metadata_map
        self.classnames = metadata_map['y']
        self.templates = templates
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.classifier = nn.Linear(model.visual.output_dim, len(self.classnames))

    def zeroshot_classifier(self, classnames, avg=True):
        """
        Zero-shot classifier for pre-trained CLIP models.
        """
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in self.templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = self.model.encode_text(texts) #embed with text encoder
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if not avg:
                zeroshot_weights.append(class_embeddings)
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def featurizer(self, input):
        input_features = self.model.encode_image(input)
        input_features = input_features / input_features.norm(dim=-1, keepdim=True)
        return input_features

    def forward(self, input, return_features=False):
        input_features = self.featurizer(input)
        logits = self.classifier(input_features.float())
        if return_features:
            return logits, input_features
        else:
            return logits

    def train(self, mode=True):
        self.model.train(mode)
        self.classifier.train(mode)
