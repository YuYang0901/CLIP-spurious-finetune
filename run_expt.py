import argparse
import os
from collections import defaultdict

import torch

try:
    import wandb
except Exception as e:
    pass

import torch.multiprocessing
import wilds

import configs.supported as supported
from configs.utils import ParseKwargs, parse_bool, populate_defaults

''' Arg defaults are filled in according to examples/configs/ '''
parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets+['imagenet'], required=True)
parser.add_argument('--num_classes', type=int, default=10, help='Maximum number of classes to have')
parser.add_argument('--imagenet_class', type=str, default='baby pacifier', help='If use ImageNet-Spurious dataset, which ImageNet class to use.')
parser.add_argument('--bingeval', default=False, type=parse_bool, const=True, nargs='?',
                    help='If true, uses Bing to search test images.')
parser.add_argument('--commercial', default=False, type=parse_bool, const=True, nargs='?',
                    help='If true, uses Bing with the commercial license.')
parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
parser.add_argument('--root_dir', required=True,
                    help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

# Dataset
parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                    help='If true, tries to download the dataset if it does not exist in root_dir.')
parser.add_argument('--frac', type=float, default=1.0,
                    help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')

# Loaders
parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
parser.add_argument('--train_loader', choices=['standard', 'group'])
parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?', help='If true, sample examples such that batches are uniform over groups.')
parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
parser.add_argument('--n_groups_per_batch', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--eval_loader', choices=['standard'], default='standard')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of batches to process before stepping optimizer and schedulers. If > 1, we simulate having a larger effective batch size (though batchnorm behaves differently).')

# Model
parser.add_argument('--model', choices=supported.models, default='clip-rn50')
parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for model initialization passed as key1=value1 key2=value2')
parser.add_argument('--noisystudent_add_dropout', type=parse_bool, const=True, nargs='?', help='If true, adds a dropout layer to the student model of NoisyStudent.')
parser.add_argument('--noisystudent_dropout_rate', type=float)
parser.add_argument('--pretrained_model_path', default=None, type=str, help='Specify a path to pretrained model weights')
parser.add_argument('--load_featurizer_only', default=False, type=parse_bool, const=True, nargs='?', help='If true, only loads the featurizer weights and not the classifier weights.')

# Transforms
parser.add_argument('--transform', choices=supported.transforms)
parser.add_argument('--additional_train_transform', choices=supported.additional_transforms, help='Optional data augmentations to layer on top of the default transforms.')
parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
parser.add_argument('--resize_scale', type=float)
parser.add_argument('--max_token_length', type=int)
parser.add_argument('--randaugment_n', type=int, help='Number of RandAugment transformations to apply.')
parser.add_argument('--reg_preprocess', default=False, type=parse_bool, const=True, nargs='?', help='If true, use regular transforms instead of clip preprocess.')

# Objective
parser.add_argument('--loss_function', choices=supported.losses)
parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

# Algorithm
parser.add_argument('--groupby_fields', nargs='+')
parser.add_argument('--group_dro_step_size', type=float)

# Multimodal
parser.add_argument('--language_weight', type=float)
parser.add_argument('--image_weight', type=float)
parser.add_argument('--crossmodal_weight', type=float)
parser.add_argument('--pos_weight', type=float)
parser.add_argument('--neg_weight', type=float)
parser.add_argument('--domain_weight', type=float)
parser.add_argument('--spurious_weight', type=float) 
parser.add_argument('--class_weight', type=float)
parser.add_argument('--clip_weight', type=float) 
parser.add_argument('--spurious_class_weight', type=float) 
parser.add_argument('--spurious_clip_weight', type=float) 
parser.add_argument('--reweight', type=parse_bool, const=True, default=False, nargs='?')
parser.add_argument('--use_group_dro', type=parse_bool, const=True, default=False, nargs='?') 
parser.add_argument('--freeze_language', type=parse_bool, const=True, default=False, nargs='?') 
parser.add_argument('--freeze_vision', type=parse_bool, const=True, default=False, nargs='?') 
parser.add_argument('--train_projection', type=parse_bool, const=True, default=False, nargs='?') 
parser.add_argument('--finetuning', choices=['zeroshot', 'linear'], default='zeroshot')
parser.add_argument('--num_templates', type=str, default='all')
parser.add_argument('--diag_spurious', type=parse_bool, const=True, default=False, nargs='?', 
    help='Only consider corresponding spurious correlations that are on the diagonal of the similarity matrix') 
parser.add_argument('--spur_img', type=parse_bool, const=True, default=False, nargs='?', 
    help='Add spurious classes to the image loss') 

# Model selection
parser.add_argument('--val_metric')
parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

# Optimization
parser.add_argument('--n_epochs', type=int)
parser.add_argument('--optimizer', choices=supported.optimizers)
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--max_grad_norm', type=float)
parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')

# Scheduler
parser.add_argument('--scheduler', choices=supported.schedulers)
parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
parser.add_argument('--scheduler_metric_name')

# Evaluation
parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
parser.add_argument('--eval_splits', nargs='+', default=[])
parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

# Misc
parser.add_argument('--device', type=int, nargs='+', default=[0])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_dir', default='./logs')
parser.add_argument('--log_every', default=50, type=int)
parser.add_argument('--save_step', type=int)
parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')

# Weights & Biases
parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
parser.add_argument('--wandb_api_key_path', type=str,
                    help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

config = parser.parse_args()
config = populate_defaults(config)

# Set device
if len(config.device) > 0:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        device_str = ",".join(map(str, config.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        device_count = torch.cuda.device_count()
        if len(config.device) > device_count:
            raise ValueError(f"Specified {len(config.device)} devices, but only {device_count} devices found.")

    config.use_data_parallel = len(config.device) > 1
    config.device = torch.device("cuda")
else:
    config.use_data_parallel = False
    config.device = torch.device("cpu")

import clip
# Necessary for large images of GlobalWheat
from PIL import ImageFile
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds.common.grouper import CombinatorialGrouper

from algorithms.initializer import initialize_algorithm
from imagenet import ImageNet
from models.initializer import get_dataset
from train import evaluate, train
from transforms import initialize_transform
from utils import (BatchLogger, Logger, get_model_prefix, initialize_wandb,
                   load, log_config, log_group_data, move_to, set_seed)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():

    # Initialize logs
    config.log_dir = f'./logs/{config.model}_{config.dataset}'
    if config.dataset == 'imagenet':
        class_name = ''.join(config.imagenet_class.split(' '))
        config.log_dir += f'-{class_name}'
        if config.bingeval:
            config.log_dir += '-bingeval'
            if config.commercial:
                config.log_dir += '-commercial'
    if (not config.eval_only) or (config.eval_epoch is None) or (config.eval_epoch >= 0):
        if config.algorithm == 'Multimodal':
            if (config.clip_weight > 0.) and (config.language_weight == 0.) and (config.image_weight == 0) and (config.spurious_weight == 0) and (config.spurious_class_weight == 0):
                config.log_dir += f'/CLIP'
            else:
                config.log_dir += f'/{config.algorithm}'
                config.log_dir += f'clip_{config.clip_weight:.1f}_lang_{config.language_weight:.1f}_spur_{config.spurious_weight:.1f}_spurclass_{config.spurious_class_weight:.1f}_spurclipclass_{config.spurious_clip_weight:.1f}_pos_{config.pos_weight:.1f}_neg_{config.neg_weight:.1f}_image_{config.image_weight:.1f}_cross_{config.crossmodal_weight:.1f}_domain_{config.domain_weight:.1f}'
                config.log_dir += f'_{config.num_templates}temp'
                if config.diag_spurious:
                    config.log_dir += '_diag'
                if config.spur_img:
                    config.log_dir += '_spurimg'
                if config.use_group_dro:
                    config.log_dir += '_groupdro'
        else:
            config.log_dir += f'/{config.algorithm}'
        if config.freeze_language:
            config.log_dir += '_freeze-language'
        if config.freeze_vision:
            config.log_dir += '_freeze-vision'
        if config.train_projection:
            config.log_dir += '_train-projection'
        if config.finetuning != 'zeroshot':
            config.log_dir += f'_{config.finetuning}'
        config.log_dir += f'_lr_{config.lr:.0e}'
        config.log_dir += f'_wd_{config.weight_decay:.0e}'        
        if config.scheduler is not None:
            config.log_dir += f'_{config.scheduler}'
        config.log_dir += f'_batchsize_{config.batch_size}'
        config.log_dir += f'_seed_{config.seed}'
    else:
        config.log_dir += f'/eval'
    if os.path.exists(config.log_dir) and config.resume:
        resume=True
        mode='a'
    elif os.path.exists(config.log_dir) and config.eval_only:
        resume=False
        mode='a'
    else:
        resume=False
        mode='w'

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    logger = Logger(os.path.join(config.log_dir, 'log.txt'), mode)

    # Record config
    log_config(config, logger)

    # Set random seed
    set_seed(config.seed)

    # Data
    full_dataset = get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        imagenet_class=config.imagenet_class,
        seed=config.seed,
        bingeval=config.bingeval,
        commercial=config.commercial,
        **config.dataset_kwargs)

    # Transforms & data augmentations for labeled dataset
    # To modify data augmentation, modify the following code block.
    # If you want to use transforms that modify both `x` and `y`,
    # set `do_transform_y` to True when initializing the `WILDSSubset` below.
    if 'clip' in config.model and not config.reg_preprocess:
        if config.model == 'clip-rn50':
            _, preprocess = clip.load('RN50')
        else:
            _, preprocess = clip.load('ViT-L/14@336px')
        train_transform = preprocess
        eval_transform = preprocess
    else:
        train_transform = initialize_transform(
            transform_name=config.transform,
            config=config,
            dataset=full_dataset,
            additional_transform_name=config.additional_train_transform,
            is_training=True)
        eval_transform = initialize_transform(
            transform_name=config.transform,
            config=config,
            dataset=full_dataset,
            is_training=False)

    train_grouper = CombinatorialGrouper(
        dataset=full_dataset,
        groupby_fields=config.groupby_fields
    )

    # Configure labeled torch datasets (WILDS dataset splits)
    datasets = defaultdict(dict)
    for split in full_dataset.split_dict.keys():
        if split=='train':
            transform = train_transform
            verbose = True
        elif split == 'val':
            transform = eval_transform
            verbose = True
        else:
            transform = eval_transform
            verbose = False
        # Get subset
        datasets[split]['dataset'] = full_dataset.get_subset(
            split,
            frac=config.frac,
            transform=transform)

        if split == 'train':
            datasets[split]['loader'] = get_train_loader(
                loader=config.train_loader,
                dataset=datasets[split]['dataset'],
                batch_size=config.batch_size,
                uniform_over_groups=config.uniform_over_groups,
                grouper=train_grouper,
                distinct_groups=config.distinct_groups,
                n_groups_per_batch=config.n_groups_per_batch,
                **config.loader_kwargs)
        else:
            datasets[split]['loader'] = get_eval_loader(
                loader=config.eval_loader,
                dataset=datasets[split]['dataset'],
                grouper=train_grouper,
                batch_size=config.batch_size,
                **config.loader_kwargs)

        # Set fields
        datasets[split]['split'] = split
        datasets[split]['name'] = full_dataset.split_names[split]
        datasets[split]['verbose'] = verbose

        # Loggers
        datasets[split]['eval_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_eval.csv'), mode=mode, use_wandb=config.use_wandb
        )
        datasets[split]['algo_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_algo.csv'), mode=mode, use_wandb=config.use_wandb
        )

    if config.use_wandb:
        initialize_wandb(config)

    # Logging dataset info
    # Show class breakdown if feasible
    if config.no_group_logging and full_dataset.is_classification and full_dataset.y_size==1 and full_dataset.n_classes <= 10:
        log_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=['y'])
    elif config.no_group_logging:
        log_grouper = None
    else:
        log_grouper = train_grouper
    log_group_data(datasets, log_grouper, logger)

    # Initialize algorithm & load pretrained weights if provided
    algorithm = initialize_algorithm(
        config=config,
        datasets=datasets,
        train_grouper=train_grouper,
    )
    # if config.use_wandb:
    #     wandb.watch(algorithm.model, log='gradients', log_freq=1)

    model_prefix = get_model_prefix(datasets['train'], config)
    if not config.eval_only:
        # Resume from most recent model in log_dir
        resume_success = False
        if resume:
            save_path = model_prefix + 'epoch:last_model.pth'
            if not os.path.exists(save_path):
                epochs = [
                    int(file.split('epoch:')[1].split('_')[0])
                    for file in os.listdir(config.log_dir) if file.endswith('.pth')]
                if len(epochs) > 0:
                    latest_epoch = max(epochs)
                    save_path = model_prefix + f'epoch:{latest_epoch}_model.pth'
            try:
                prev_epoch, best_val_metric = load(algorithm, save_path, device=config.device)
                epoch_offset = prev_epoch + 1
                logger.write(f'Resuming from epoch {epoch_offset} with best val metric {best_val_metric}')
                resume_success = True
            except FileNotFoundError:
                pass
        if resume_success == False:
            epoch_offset=0
            best_val_metric=None

        # Log effective batch size
        if config.gradient_accumulation_steps > 1:
            logger.write(
                (f'\nUsing gradient_accumulation_steps {config.gradient_accumulation_steps} means that')
                + (f' the effective labeled batch size is {config.batch_size * config.gradient_accumulation_steps}')
                + ('. Updates behave as if torch loaders have drop_last=False\n')
            )

        train(
            algorithm=algorithm,
            datasets=datasets,
            general_logger=logger,
            config=config,
            epoch_offset=epoch_offset,
            best_val_metric=best_val_metric,
        )
    else:
        if (config.eval_epoch is None) or (config.eval_epoch >= 0):
            if config.eval_epoch is None:
                eval_model_path = model_prefix + 'epoch:best_model.pth'
            elif (config.eval_epoch == config.n_epochs):
                eval_model_path = model_prefix + 'epoch:last_model.pth'
            else:
                eval_model_path = model_prefix +  f'epoch:{config.eval_epoch}_model.pth'
            print(eval_model_path)
            best_epoch, best_val_metric = load(algorithm, eval_model_path, device=config.device)
            if config.eval_epoch is None:
                epoch = best_epoch
            else:
                epoch = config.eval_epoch
            if epoch == best_epoch:
                is_best = True
            else:
                is_best = False
        else:
            epoch = config.eval_epoch
            is_best = False

        evaluate(
            algorithm=algorithm,
            datasets=datasets,
            epoch=epoch,
            general_logger=logger,
            config=config,
            is_best=is_best)

    if config.use_wandb:
        wandb.finish()
    logger.close()
    for split in datasets:      
        datasets[split]['eval_logger'].close()
        datasets[split]['algo_logger'].close()

if __name__=='__main__':
    main()