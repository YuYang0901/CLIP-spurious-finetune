import torch
import torch.nn as nn
from wilds.common.metrics.loss import ElementwiseLoss, MultiTaskLoss
from wilds.common.metrics.all_metrics import MSE
from utils import cross_entropy_with_logits_loss
from wilds.common.metrics.metric import ElementwiseMetric
from wilds.common.utils import maximum, avg_over_groups, numel

import wandb
import matplotlib.pyplot as plt
import numpy as np

class MultimodalLoss(ElementwiseMetric):
    def __init__(self, loss_fn, config, name=None):
        self.loss_fn = loss_fn
        self.sim_fun = nn.CosineSimilarity(dim=2, eps=1e-6)

        self.language_weight = config.language_weight
        self.image_weight = config.image_weight
        self.crossmodal_weight = config.crossmodal_weight
        self.pos_weight = config.pos_weight
        self.neg_weight = config.neg_weight
        self.domain_weight = config.domain_weight
        self.spurious_weight = config.spurious_weight
        self.class_weight = config.class_weight
        self.clip_weight = config.clip_weight
        self.spurious_class_weight = config.spurious_class_weight
        self.spurious_clip_weight = config.spurious_clip_weight

        self.save_dir = config.log_dir
        self.use_wandb = config.use_wandb
        self.diag_spurious = config.diag_spurious
        self.spur_img = config.spur_img

        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def similarity_loss(self, features_1, features_2, labels_1, labels_2, sep_pos_neg=False, abs=True):
        loss = self.sim_fun(features_1.unsqueeze(1), features_2.unsqueeze(0))

        pos_mask = (labels_1.unsqueeze(1).expand(-1, len(labels_2))-labels_2.unsqueeze(0).expand(len(labels_1),-1)) == 0
        pos_mask = pos_mask.float().cuda()

        pos_loss = (loss * pos_mask).mean(dim=-1)
        neg_loss = (loss * (1- pos_mask)).mean(dim=-1)
        if abs:
            neg_loss = neg_loss.abs()

        if sep_pos_neg:
            return (pos_loss, neg_loss)
        else:
            return - self.pos_weight * pos_loss + self.neg_weight * neg_loss

    def _compute(self, results):
        element_wise_metrics,losses = self._compute_element_wise(results)
        avg_metric = element_wise_metrics.mean()
        return avg_metric, losses

    def compute(self, results, return_dict=True):
        if numel(results['y_true']) == 0:
            if hasattr(results['y_true'], 'device'):
                agg_metric = torch.tensor(0., device=results['y_true'].device)
            else:
                agg_metric = torch.tensor(0.)
        else:
            agg_metric, losses = self._compute(results)
        if return_dict:
            results = {
                self.agg_metric_field: agg_metric.item()
            }
            return results, losses
        else:
            return agg_metric, losses

    def _compute_element_wise(self, results):
        # Multimodal classification CE loss
        classification_loss = self.loss_fn(results['y_pred'], results['y_true'])
        
        language_loss = torch.zeros_like(classification_loss)
        language_loss_pos = torch.zeros_like(classification_loss)
        language_loss_neg = torch.zeros_like(classification_loss)
        image_loss = torch.zeros_like(classification_loss)
        crossmodal_loss = torch.zeros_like(classification_loss)
        spurious_loss = torch.zeros_like(classification_loss)
        spurious_classification_loss = torch.zeros_like(classification_loss)

        features = results["features"]
        embeddings = results["embeddings"]
        metadata = results["metadata"]
        fields = embeddings.keys()
        
        # CLIP multimodal contrastive CE loss
        logits_per_image = results["logits"]
        logits_per_text = logits_per_image.t()
        targets = torch.arange(len(logits_per_image),dtype=torch.long).cuda()
        image_ce = self.loss_fn(logits_per_image, targets)
        language_ce = self.loss_fn(logits_per_text, targets)
        clip_loss = .5 * (image_ce + language_ce)

        if self.spurious_clip_weight > 0.:
            logits_per_image = results["spur_logits"]
            logits_per_text = logits_per_image.t()
            targets = torch.arange(len(logits_per_image),dtype=torch.long).cuda()
            spurious_classification_loss = spurious_classification_loss + .5 * (self.loss_fn(logits_per_image, targets) + self.loss_fn(logits_per_text, targets))

        # Language features
        y_embeddings = embeddings['y']
        num_templates, num_classes, feat_dim = y_embeddings.shape
        y_embedding_labels = torch.repeat_interleave(torch.arange(num_classes), num_templates)
        y_embeddings = y_embeddings.view(-1, feat_dim)           
            
        for field_idx, field in enumerate(fields):

            if field == 'y':
                if self.image_weight > 0.:
                    # Image in-modal loss
                    image_loss = image_loss + self.similarity_loss(features, features, metadata[:, field_idx], metadata[:, field_idx], abs=False)

                # Language in-modal loss
                if self.language_weight > 0.:
                    loss_pos, loss_neg = self.similarity_loss(y_embeddings, y_embeddings, y_embedding_labels, y_embedding_labels, sep_pos_neg=True, abs=False)
                    language_loss_pos = language_loss_pos - loss_pos[results['y_true']]
                    language_loss_neg = language_loss_neg + loss_neg[results['y_true']]
                    loss = - self.pos_weight * loss_pos + self.neg_weight * loss_neg
                    loss = loss.view(num_templates, num_classes).mean(-1)[results['y_true']]
                    language_loss = language_loss + loss

                # Cross-modal loss
                if self.crossmodal_weight > 0.:                        
                    crossmodal_loss = crossmodal_loss + self.similarity_loss(features, y_embeddings, metadata[:, field_idx], y_embedding_labels, abs=False)
            
            
            if field == 'generic-spurious':

                # Spurious classification CE loss
                if self.spurious_class_weight > 0.:
                    spurious_classification_loss = spurious_classification_loss + self.loss_fn(results['spur_pred'][field], results['spur_true'][field])

                if self.domain_weight > 0:

                    # Language features
                    field_embeddings = embeddings[field]
                    embedding_labels = torch.repeat_interleave(torch.arange(field_embeddings.shape[1]), num_templates)
                    field_embeddings = field_embeddings.view(-1, feat_dim)

                    # Language in-modal loss
                    if self.language_weight > 0.:
                        loss_pos, loss_neg = self.similarity_loss(field_embeddings, field_embeddings, embedding_labels, embedding_labels, sep_pos_neg=True, abs=False)
                        language_loss_pos = language_loss_pos - loss_pos[metadata[:, field_idx]]
                        language_loss_neg = language_loss_neg + loss_neg[metadata[:, field_idx]]
                        loss = - self.pos_weight * loss_pos + self.neg_weight * loss_neg
                        loss = loss.view(num_templates, num_classes).mean(-1)[metadata[:, field_idx]]
                        language_loss = language_loss + self.domain_weight * loss
                        spurious_loss = spurious_loss + self.language_weight * loss

            elif field == 'spurious' and self.spurious_weight > 0 or self.use_wandb:
                if self.image_weight > 0. and self.spur_img:
                    # Image in-modal loss
                    image_loss = image_loss + self.similarity_loss(features, features, metadata[:, field_idx], metadata[:, field_idx], abs=False)

                # Language features
                field_embeddings = embeddings[field]
                embedding_labels = torch.repeat_interleave(torch.arange(field_embeddings.shape[1]), num_templates)
                field_embeddings = field_embeddings.view(-1, feat_dim)

                # Language spurious loss
                if self.language_weight > 0. or self.use_wandb:
                    sim = self.sim_fun(y_embeddings.unsqueeze(1), field_embeddings.unsqueeze(0))
                    descriptions = results['descriptions']
                    if self.diag_spurious:
                        loss = self.neg_weight * torch.diagonal(sim).abs()
                    else:
                        loss = self.neg_weight * sim.abs().mean(dim=-1)
                    loss = loss.view(num_templates, num_classes).mean(-1)[results['y_true']]
                    language_loss = language_loss + self.spurious_weight * loss
                    spurious_loss = spurious_loss + self.language_weight * loss

                # Cross-modal spurious loss
                if self.crossmodal_weight > 0.:
                    sim = self.sim_fun(features.unsqueeze(1), field_embeddings.unsqueeze(0))                      
                    if self.diag_spurious:
                        loss = self.neg_weight * torch.diagonal(sim).abs()
                    else:
                        loss = self.neg_weight * sim.abs().mean(dim=-1)
                    crossmodal_loss = crossmodal_loss + self.spurious_weight * loss
                    spurious_loss = spurious_loss + self.crossmodal_weight * loss

        loss = self.clip_weight * clip_loss + self.class_weight * classification_loss + self.language_weight * language_loss + self.image_weight * image_loss + self.crossmodal_weight * crossmodal_loss + (self.spurious_class_weight+self.spurious_clip_weight) * spurious_classification_loss
        
        return loss, (classification_loss, image_ce, language_ce, language_loss, language_loss_pos, language_loss_neg, image_loss, crossmodal_loss, spurious_loss, spurious_classification_loss)

    def compute_group_wise(self, results, g, n_groups, return_dict=True):
        group_metrics, group_counts, worst_group_metric, losses = self._compute_group_wise(results, g, n_groups)
        if return_dict:
            results = {}
            for group_idx in range(n_groups):
                results[self.group_metric_field(group_idx)] = group_metrics[group_idx].item()
                results[self.group_count_field(group_idx)] = group_counts[group_idx].item()
            results[self.worst_group_metric_field] = worst_group_metric.item()
            return results, losses
        else:
            return group_metrics, group_counts, worst_group_metric, losses

    def _compute_group_wise(self, results, g, n_groups):
        element_wise_metrics, losses = self._compute_element_wise(results)
        group_metrics, group_counts = avg_over_groups(element_wise_metrics, g, n_groups)
        worst_group_metric = self.worst(group_metrics[group_counts>0])

        avg_loss = []
        for loss in losses:
            group_metrics_, _ = avg_over_groups(loss, g, n_groups)
            avg_loss.append(group_metrics_)
        return group_metrics, group_counts, worst_group_metric, avg_loss

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)


def initialize_loss(loss, config):
    if loss == 'cross_entropy':
        if config.algorithm != 'Multimodal':
            return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
        else:
            return MultimodalLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100), config=config)

    elif loss == 'lm_cross_entropy':
        return MultiTaskLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))

    elif loss == 'mse':
        return MSE(name='loss')

    elif loss == 'multitask_bce':
        return MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'))

    elif loss == 'cross_entropy_logits':
        return ElementwiseLoss(loss_fn=cross_entropy_with_logits_loss)

    else:
        raise ValueError(f'loss {loss} not recognized')