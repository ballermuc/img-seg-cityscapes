import segmentation_models_pytorch as smp
import sys
from tqdm import tqdm
import os
import cv2
import torch
import lookup_table as lut
import wandb


class Epoch:
    def __init__(
        self,
        model,
        loss,
        optimizer=None,
        s_phase="test",
        p_dir_export=None,
        device="cpu",
        verbose=True,
        writer=None,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        if s_phase not in ["training", "validation", "test"]:
            raise ValueError(
                f'Incorrect value for s_phase: "{s_phase}"\n'
                f'Please use one of: "training", "validation", "test"'
            )
        self.s_phase = s_phase
        self.p_dir_export = p_dir_export
        self.device = device
        self.verbose = verbose
        self.writer = writer
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, image, target=None):
        if self.s_phase == "training":
            self.optimizer.zero_grad()
            prediction = self.model.forward(image)
            loss = self.loss(prediction, target)
            loss.backward()
            self.optimizer.step()
            return loss, prediction
        elif self.s_phase == "validation":
            with torch.inference_mode():
                prediction = self.model.forward(image)
                loss = self.loss(prediction, target)
            return loss, prediction
        else:  # assume "test"
            with torch.inference_mode():
                prediction = self.model.forward(image)
            return None, prediction

    def on_epoch_start(self):
        if self.s_phase == "training":
            self.model.train()
        else:  # assume "validation" or "test"
            self.model.eval()

    def run(self, dataloader, i_epoch=-1):
        self.on_epoch_start()
        logs = {}
        n_iteration_sum = 0
        l_loss_sum = 0
        d_confusion = {"tp": None, "fp": None, "fn": None, "tn": None}

        with tqdm(dataloader, desc=self.s_phase, file=sys.stdout, disable=not self.verbose) as iterator:
            for image, target, l_p_image, l_p_target in iterator:
                n_iteration_sum += 1
                image = image.to(self.device)
                if self.s_phase != "test":
                    target = target.unsqueeze(dim=1).to(dataloader.dataset.device)
                    target = lut.lookup_nchw(
                        td_u_input=target,
                        td_i_lut=dataloader.dataset.th_i_lut_id2trainid,
                    )
                    target.squeeze_(dim=1)
                    target = target.long().to(self.device)
                    loss, logits = self.batch_update(image, target)
                    prediction = logits.argmax(axis=1, keepdim=True)

                    # Update loss logs
                    loss_value = loss.cpu().detach().numpy()
                    l_loss_sum += loss_value
                    loss_logs = {self.loss.__name__: l_loss_sum / n_iteration_sum}
                    logs.update(loss_logs)

                    # Update confusion matrix
                    tp, fp, fn, tn = smp.metrics.get_stats(
                        prediction.squeeze(dim=1),
                        target,
                        mode="multiclass",
                        num_classes=19,
                        ignore_index=19,
                    )
                    if d_confusion["tp"] is None:
                        d_confusion["tp"] = tp.sum(dim=0, keepdim=True)
                    else:
                        d_confusion["tp"] += tp.sum(dim=0, keepdim=True)
                    if d_confusion["fp"] is None:
                        d_confusion["fp"] = fp.sum(dim=0, keepdim=True)
                    else:
                        d_confusion["fp"] += fp.sum(dim=0, keepdim=True)
                    if d_confusion["fn"] is None:
                        d_confusion["fn"] = fn.sum(dim=0, keepdim=True)
                    else:
                        d_confusion["fn"] += fn.sum(dim=0, keepdim=True)
                    if d_confusion["tn"] is None:
                        d_confusion["tn"] = tn.sum(dim=0, keepdim=True)
                    else:
                        d_confusion["tn"] += tn.sum(dim=0, keepdim=True)
                else:
                    _, logits = self.batch_update(image)
                    prediction = logits.argmax(axis=1, keepdim=True)

            # Compute IoU metrics
            if self.s_phase != "test":
                logs["iou_score"] = smp.metrics.functional.iou_score(
                    tp=d_confusion["tp"],
                    fp=d_confusion["fp"],
                    fn=d_confusion["fn"],
                    tn=d_confusion["tn"],
                    reduction="macro-imagewise",
                ).detach().cpu().numpy()

                # Per-class IoU
                per_class_iou = smp.metrics.functional.iou_score(
                    tp=d_confusion["tp"],
                    fp=d_confusion["fp"],
                    fn=d_confusion["fn"],
                    tn=d_confusion["tn"],
                    reduction="none",
                ).detach().cpu().numpy()

                per_class_iou = per_class_iou.mean(axis=0)
                logs["per_class_iou"] = per_class_iou

                # Compute Pixel Accuracy
                logs["pixel_accuracy"] = smp.metrics.functional.accuracy(
                    tp=d_confusion["tp"],
                    fp=d_confusion["fp"],
                    fn=d_confusion["fn"],
                    tn=d_confusion["tn"],
                    reduction="macro-imagewise",
                ).detach().cpu().numpy()

                # Compute Dice Coefficient
                logs["dice_coefficient"] = smp.metrics.functional.f1_score(
                    tp=d_confusion["tp"],
                    fp=d_confusion["fp"],
                    fn=d_confusion["fn"],
                    tn=d_confusion["tn"],
                    reduction="macro-imagewise",
                ).detach().cpu().numpy()

            # Unified logging for WandB
            if self.writer is not None:
                phase = "Training" if self.s_phase == "training" else "Validation"
                log_data = {
                    f"Overall IoU/{phase}": logs.get("iou_score", 0),
                    f"Loss/{phase}": logs.get(self.loss.__name__, 0),
                    f"Pixel Accuracy/{phase}": logs.get("pixel_accuracy", 0),
                    f"Dice Coefficient/{phase}": logs.get("dice_coefficient", 0),
                    "Epoch": i_epoch,
                }

                if "per_class_iou" in logs:
                    class_names = [
                        c.name for c in dataloader.dataset.classes
                        if c.train_id not in [-1, 255]
                    ]
                    for class_idx, class_iou in enumerate(logs["per_class_iou"]):
                        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
                        log_data[f"Class IoU/{class_name}/{phase}"] = float(class_iou)

                if self.s_phase == "validation":
                    log_data.update({
                        "Predictions/Color": [
                            wandb.Image(
                                lut.lookup_nchw(
                                    td_u_input=prediction[i].unsqueeze(0).byte(),
                                    td_i_lut=dataloader.dataset.th_i_lut_trainid2color
                                ),
                                caption=f"Prediction {i}: {l_p_image[i]}"
                            ) for i in range(min(4, prediction.shape[0]))
                        ],
                        "Targets/Color": [
                            wandb.Image(
                                lut.lookup_nchw(
                                    td_u_input=target[i].unsqueeze(0).unsqueeze(dim=1).byte(),
                                    td_i_lut=dataloader.dataset.th_i_lut_trainid2color
                                ),
                                caption=f"Target {i}: {l_p_target[i]}"
                            ) for i in range(min(4, target.shape[0]))
                        ],
                        "Images/Color": [
                            wandb.Image(
                                ((image[i] + 2) * 64).round().clamp(0, 255).byte(),
                                caption=f"Image {i}: {l_p_image[i]}"
                            ) for i in range(min(4, image.shape[0]))
                        ]
                    })

                self.writer.log(log_data, step=i_epoch)

            return logs
