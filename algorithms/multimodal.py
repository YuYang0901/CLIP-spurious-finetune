import torch
from utils import update_average
from models.initializer import initialize_model
from algorithms.single_model_algorithm import SingleModelAlgorithm
from wilds.common.utils import numel


class Multimodal(SingleModelAlgorithm):
    """
    Multimodal contrastive representation learning with in-modal, cross-modal and spurious loss.
    """

    def __init__(
        self,
        config,
        grouper,
        loss,
        metric,
        n_train_steps,   
        is_group_in_train=None,
    ):
        # Initialize model
        model = initialize_model(config, d_out=None)       

        # Initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

        # Algorithm hyperparameters
        self.use_group_dro = config.use_group_dro
        self.reweight = config.reweight
        self.finetuning = config.finetuning

        # Additional logging
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("clip_loss")
        self.logged_fields.append("image_ce")
        self.logged_fields.append("language_ce")
        self.logged_fields.append("language_loss")
        self.logged_fields.append("language_loss_pos")
        self.logged_fields.append("language_loss_neg")
        self.logged_fields.append("image_loss")
        self.logged_fields.append("crossmodal_loss")
        self.logged_fields.append("spurious_loss")
        self.logged_fields.append("spurious_classification_loss")

        if self.use_group_dro:
            # additional logging
            self.logged_fields.append('group_weight')
            # step size
            self.group_weights_step_size = config.group_dro_step_size
            # initialize adversarial weights
            self.group_weights = torch.zeros(grouper.n_groups)
            self.group_weights[is_group_in_train] = 1
            self.group_weights = self.group_weights/self.group_weights.sum()
            self.group_weights = self.group_weights.to(self.device)

    def process_batch(self, batch):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - features (Tensor): featurizer output for batch
                - y_pred (Tensor): full model output for batch 
        """
        # Forward pass
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        y_pred, features = self.model(x, return_features=True)

        if self.finetuning == 'linear':
            y_pred = self.model.classifier(features.clone().detach().float())

        embeddings = {}
        descriptions = {}
        spur_true = {}
        spur_pred = {}
        try:
            for i, field in enumerate(self.model.module.metadata_map.keys()):
                embeddings[field] = self.model.module.zeroshot_classifier(self.model.module.metadata_map[field], avg=False)
                descriptions[field] = [template.format(classname) for classname in self.model.module.metadata_map[field] for template in self.model.module.templates]
                if field == 'generic-spurious':
                    spur_true[field] = metadata[:, i].to(self.device)
                    embedding = embeddings[field]
                    embedding = embedding.mean(dim=0)
                    embedding = embedding / embedding.norm()
                    spur_pred[field] = self.model.module.logit_scale.exp() * features @ embedding.T

                    zeroshot_weights = torch.mean(embeddings[field], dim=0)[spur_true[field]]
                    spur_logits = self.model.module.logit_scale.exp() * features @ zeroshot_weights.t()
                elif field == 'y':
                    zeroshot_weights = torch.mean(embeddings[field], dim=0)[y_true]
                    logits = self.model.module.logit_scale.exp() * features @ zeroshot_weights.t()
        except:
            for i, field in enumerate(self.model.metadata_map.keys()):
                embeddings[field] = self.model.zeroshot_classifier(self.model.metadata_map[field], avg=False)
                descriptions[field] = [template.format(classname) for classname in self.model.metadata_map[field] for template in self.model.templates]
                if field == 'generic-spurious':
                    spur_true[field] = metadata[:, i].to(self.device)
                    embedding = embeddings[field]
                    embedding = embedding.mean(dim=0)
                    embedding = embedding / embedding.norm()
                    spur_pred[field] = self.model.logit_scale.exp() * features @ embedding.T

                    zeroshot_weights = torch.mean(embeddings[field], dim=0)[spur_true[field]]
                    spur_logits = self.model.logit_scale.exp() * features @ zeroshot_weights.t()
                elif field == 'y':
                    zeroshot_weights = torch.mean(embeddings[field], dim=0)[y_true]
                    logits = self.model.logit_scale.exp() * features @ zeroshot_weights.t()
        results = {
            "g": g,
            "metadata": metadata,
            "y_true": y_true,
            "y_pred": y_pred,
            "spur_true": spur_true,
            "spur_pred": spur_pred,
            "features": features,
            "embeddings": embeddings,
            "logits": logits,
            "spur_logits": spur_logits,
            'descriptions': descriptions,
            "batch_idx": self.batch_idx,
            "is_training": self.is_training
        }

        if self.use_group_dro:
            results['group_weight'] = self.group_weights

        return results

    def objective(self, results):
        if self.use_group_dro:
            group_losses, _, _, losses = self.loss.compute_group_wise(
            results,
            results['g'],
            self.grouper.n_groups,
            return_dict=False)
            loss = group_losses.float() @ self.group_weights
            losses = [loss.float() @ self.group_weights for loss in losses]
        else:
            if self.reweight:
                loss, losses = self.loss._compute_element_wise(results)
                g = results['g']
                w = torch.tensor([1./torch.sum(g == gi) for gi in g]).cuda()
                loss = (loss * w).mean()
                losses = [(loss * w).mean() for loss in losses]
            else:
                loss, losses = self.loss.compute(
                    results, return_dict=False
                )
                loss = loss.mean()
                losses = [loss.mean() for loss in losses]

        (
            classification_loss, 
            image_ce,
            language_ce, 
            language_loss, 
            language_loss_pos, 
            language_loss_neg, 
            image_loss, 
            crossmodal_loss, 
            spurious_loss,
            spurious_classification_loss
        ) = losses

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(
            results, "clip_loss", .5 * (image_ce + language_ce)
        )
        self.save_metric_for_logging(
            results, "image_ce", image_ce
        )
        self.save_metric_for_logging(
            results, "language_ce", language_ce
        )
        self.save_metric_for_logging(
            results, "language_loss", language_loss
        )
        self.save_metric_for_logging(
            results, "language_loss_pos", language_loss_pos
        )
        self.save_metric_for_logging(
            results, "language_loss_neg", language_loss_neg
        )
        self.save_metric_for_logging(
            results, "image_loss", image_loss
        )
        self.save_metric_for_logging(
            results, "crossmodal_loss", crossmodal_loss
        )
        self.save_metric_for_logging(
            results, "spurious_loss", spurious_loss
        )
        self.save_metric_for_logging(
            results, "spurious_classification_loss", spurious_loss
        )
        return loss

    def _update(self, results, should_step=True):
        """
        Process the batch, update the log, and update the model, group weights, and scheduler.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
                - objective (float)
        """
        if self.use_group_dro:
            # compute group losses
            group_losses, _, _, _ = self.loss.compute_group_wise(
                results,
                results['g'],
                self.grouper.n_groups,
                return_dict=False
            )
            # update group weights
            self.group_weights = self.group_weights * torch.exp(self.group_weights_step_size*group_losses.data)
            self.group_weights = (self.group_weights/(self.group_weights.sum()))
            # save updated group weights
            results['group_weight'] = self.group_weights
        # update model
        super()._update(results, should_step=should_step)

    def update_log(self, results):
        """
        Updates the internal log, Algorithm.log_dict
        Args:
            - results (dictionary)
        """
        results = self.sanitize_dict(results, to_out_device=False)
        # check all the fields exist
        for field in self.logged_fields:
            assert field in results, f"field {field} missing"
        # compute statistics for the current batch
        batch_log = {}
        with torch.no_grad():
            for m in self.logged_metrics:
                if not self.no_group_logging:
                    if m.name == 'loss':
                        group_metrics, group_counts, _, _ = m.compute_group_wise(
                            results,
                            g=results['g'],
                            n_groups=self.grouper.n_groups,
                            return_dict=False)
                    else:
                        group_metrics, group_counts, worst_group_metric = m.compute_group_wise(
                        results['y_pred'],
                        results['y_true'],
                        results['g'],
                        self.grouper.n_groups,
                        return_dict=False)
                    batch_log[f'{self.group_prefix}{m.name}'] = group_metrics
                try:
                    loss, _ = m.compute(
                        results,
                        return_dict=False)
                    batch_log[m.agg_metric_field] = loss.item()
                except:
                    batch_log[m.agg_metric_field] = m.compute(
                    results['y_pred'],
                    results['y_true'],
                    return_dict=False).item()
                
            count = numel(results['y_true'])

        # transfer other statistics in the results dictionary
        for field in self.logged_fields:
            if field.startswith(self.group_prefix) and self.no_group_logging:
                continue
            v = results[field]
            if isinstance(v, torch.Tensor) and v.numel()==1:
                batch_log[field] = v.item()
            else:
                if isinstance(v, torch.Tensor):
                    assert v.numel()==self.grouper.n_groups, "Current implementation deals only with group-wise statistics or a single-number statistic"
                    assert field.startswith(self.group_prefix)
                batch_log[field] = v

        # update the log dict with the current batch
        if not self._has_log: # since it is the first log entry, just save the current log
            self.log_dict = batch_log
            if not self.no_group_logging:
                self.log_dict[self.group_count_field] = group_counts
            self.log_dict[self.count_field] = count
        else: # take a running average across batches otherwise
            for k, v in batch_log.items():
                if k.startswith(self.group_prefix):
                    if self.no_group_logging:
                        continue
                    self.log_dict[k] = update_average(self.log_dict[k], self.log_dict[self.group_count_field], v, group_counts)
                else:
                    self.log_dict[k] = update_average(self.log_dict[k], self.log_dict[self.count_field], v, count)
            if not self.no_group_logging:
                self.log_dict[self.group_count_field] += group_counts
            self.log_dict[self.count_field] += count
        self._has_log = True