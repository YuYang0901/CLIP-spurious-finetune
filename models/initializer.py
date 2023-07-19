import clip
import torch.nn as nn

from .clip import (ImageEncoder, LinearProbCLIP, ZeroShotClassifier,
                   imagenet_templates)


def get_dataset(dataset, version=None, imagenet_class='baby pacifier', seed=42, bingeval=False, commercial=False, **dataset_kwargs):
    """
    Returns the appropriate WILDS dataset class.
    Input:
        dataset (str): Name of the dataset
        version (Union[str, None]): Dataset version number, e.g., '1.0'.
                                    Defaults to the latest version.
        dataset_kwargs: Other keyword arguments to pass to the dataset constructors.
    Output:
        The specified WILDSDataset class.
    """
    if version is not None:
        version = str(version)

    if dataset == 'waterbirds':
        from waterbirds_dataset import WaterbirdsDataset
        return WaterbirdsDataset(version=version, **dataset_kwargs)
    elif dataset == 'imagenet':
        from imagenet import ImagenetSpurious
        return ImagenetSpurious(version=version, imagenet_class=imagenet_class, bingeval=bingeval, commercial=commercial, seed=seed, **dataset_kwargs)

def initialize_model(config, d_out, is_featurizer=False, metadata_map=None):
    """
    Initializes models according to the config
        Args:
            - config (dictionary): config dictionary
            - d_out (int): the dimensionality of the model output
            - is_featurizer (bool): whether to return a model or a (featurizer, classifier) pair that constitutes a model.
        Output:
            If is_featurizer=True:
            - featurizer: a model that outputs feature Tensors of shape (batch_size, ..., feature dimensionality)
            - classifier: a model that takes in feature Tensors and outputs predictions. In most cases, this is a linear layer.

            If is_featurizer=False:
            - model: a model that is equivalent to nn.Sequential(featurizer, classifier)

        Pretrained weights are loaded according to config.pretrained_model_path using either transformers.from_pretrained (for bert-based models)
        or our own utils.load function (for torchvision models, resnet18-ms, and gin-virtual).
        There is currently no support for loading pretrained weights from disk for other models.
    """
    # If load_featurizer_only is True,
    # then split into (featurizer, classifier) for the purposes of loading only the featurizer,
    # before recombining them at the end
    featurize = is_featurizer or config.load_featurizer_only

    if 'clip' in config.model:
        if featurize:
            model = initialize_clip_model(config, featurize=True)
        else:
            model = initialize_clip_model(config)

    else:
        raise ValueError(f'Model: {config.model} not recognized.')

    # Recombine model if we originally split it up just for loading
    if featurize and not is_featurizer:
        model = nn.Sequential(*model)

    # The `needs_y` attribute specifies whether the model's forward function
    # needs to take in both (x, y).
    # If False, Algorithm.process_batch will call model(x).
    # If True, Algorithm.process_batch() will call model(x, y) during training,
    # and model(x, None) during eval.
    if not hasattr(model, 'needs_y'):
        # Sometimes model is a tuple of (featurizer, classifier)
        if is_featurizer:
            for submodel in model:
                if submodel is not None:
                    submodel.needs_y = False
        else:
            model.needs_y = False

    return model


def initialize_clip_model(config, featurize=False):
    from models.clip import ZeroShotCLIP
    dataset = get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        imagenet_class=config.imagenet_class,
        seed=config.seed,
        **config.dataset_kwargs)
    dataset.metadata_map.pop('from_source_domain')
    
    if config.model == 'clip-rn50':
        featurizer, _ = clip.load('RN50')
        proj_name = 'attnpool'
    elif config.model == 'clip-vit':
        featurizer, _ = clip.load('ViT-L/14@336px')
    else:
        raise ValueError(f'Model: {config.model} not recognized.')

    if config.freeze_vision:
        for param in featurizer.visual.parameters():
            param.requires_grad = False
    if config.freeze_language:
        for param in featurizer.transformer.parameters():
            param.requires_grad = False
    if config.train_projection:
        if config.model == 'clip-rn50':
            for param in featurizer.visual._modules[proj_name].parameters():
                param.requires_grad = True
        elif config.model == 'clip-vit':
            featurizer.visual.proj.requires_grad = True

    if config.num_templates == 'all':
        templates = imagenet_templates
    elif config.num_templates == '2':
        templates = ['a photo of the {}.', 'a photo of a {}.',]
    else:
        templates = imagenet_templates[:int(config.num_templates)]
    
    if featurize:
        if config.finetuning == 'zeroshot':
            return ImageEncoder(featurizer), ZeroShotClassifier(featurizer, dataset.metadata_map, templates=templates)
        else:
            return ImageEncoder(featurizer), nn.Linear(featurizer.visual.output_dim, len(dataset.metadata_map['y']))
    else:
        if config.finetuning == 'zeroshot':
            model = ZeroShotCLIP(featurizer, dataset.metadata_map, templates=templates)
        else:
            model = LinearProbCLIP(featurizer, dataset.metadata_map, templates=templates)
        return model
