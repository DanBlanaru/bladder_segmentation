from logging import root
from monai.transforms.croppad.dictionary import DivisiblePadD
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, checkpoint
from pytorch_lightning.loggers import WandbLogger
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
    DivisiblePadd
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset, pad_list_data_collate
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import tempfile
import shutil
import os
import glob
from utils import AMOSDataset


print_config()

# root_dir = ("/data/dan_blanaru/merged_AMOS_CTORG/")
root_dir = ("/data/dan_blanaru/AMOS22_preprocessed/")
makeshift_log = open(os.path.join(root_dir,'makeshift_log.csv'),'w')

print(root_dir)


hparams = {
    "max_epochs":200,
    "dataset_name":root_dir,
    "batch_size":4,
    "lr" : 1e-4,
    "using_rand_crop" : False,

}

class Net(pytorch_lightning.LightningModule):
    def __init__(self,max_epochs,dataset_name,batch_size,lr,using_rand_crop):
        super().__init__()
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.max_epochs = max_epochs
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.lr = lr
        self.using_rand_crop = using_rand_crop
        self.save_hyperparameters()

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):

        set_determinism(seed=0)

        # define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                DivisiblePadd(["image", "label"], 16),
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(1.5, 1.5, 2.0),
                #     mode=("bilinear", "nearest"),
                # ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # randomly crop out patch samples from
                # big image based on pos / neg ratio
                # the image centers of negative samples
                # must be in valid image area
                # RandCropByPosNegLabeld(
                #     keys=["image", "label"],
                #     label_key="label",
                #     spatial_size=(96, 96, 96),
                #     pos=1,
                #     neg=1,
                #     num_samples=4,
                #     image_key="image",
                #     image_threshold=0,
                # ),
                # user can also add other random transforms
                #                 RandAffined(
                #                     keys=['image', 'label'],
                #                     mode=('bilinear', 'nearest'),
                #                     prob=1.0,
                #                     spatial_size=(96, 96, 96),
                #                     rotate_range=(0, 0, np.pi/15),
                #                     scale_range=(0.1, 0.1, 0.1)),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                DivisiblePadd(["image", "label"], 16),
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(1.5, 1.5, 2.0),
                #     mode=("bilinear", "nearest"),
                # ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )

        print(f"Using {self.dataset_name}")
        data_dir =self.dataset_name
        train_images = sorted(
            glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        train_labels = sorted(
            glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]
        
        train_files, val_files = data_dicts[10:160], data_dicts[160:]
        print(f"{len(train_files)} test items, {len(val_files)} validation items out of {len(data_dicts)} total")
        # cache datasets do not fit into memory with my setup
        # self.train_ds = CacheDataset(
        #     data=train_files, transform=train_transforms,
        #     cache_rate=1, num_workers=4,
        # )
        # self.val_ds = CacheDataset(
        #     data=val_files, transform=val_transforms,
        #     cache_rate=1, num_workers=4,
        # )

        self.train_ds = Dataset(
            data=train_files, transform=train_transforms)
        self.val_ds = Dataset(
            data=val_files, transform=val_transforms)

        # self.train_ds = AMOSDataset(json_path="toy_dataset.json",root_dir="/data/dan_blanaru/AMOS22_preprocessed/", transform=train_transforms,train_size=8,is_val=False)
        # self.val_ds = AMOSDataset(json_path="toy_dataset.json",root_dir="/data/dan_blanaru/AMOS22_preprocessed/", transform=val_transforms,train_size=8,is_val=True)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=4, collate_fn=pad_list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=2,
            collate_fn=pad_list_data_collate)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        
        images, labels = batch["image"], batch["label"]
        
        output = self.forward(images)
        loss = self.loss_function(output, labels)

        outputs = [self.post_pred(i) for i in decollate_batch(output)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred = outputs,y = labels)
        mean_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        
        tensorboard_logs = {"train_loss": loss.item()}
        self.log("train_dice", mean_dice)
        self.log("train_loss", loss.item())
        print("train: ",loss)
        
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        print("val loss",loss)
        self.log("val_loss",loss)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_loss": loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.log("val_dice", mean_val_dice)
        self.log("val_loss", mean_val_loss)
        return {"log": tensorboard_logs}



# initialise the LightningModule
net = Net(**hparams)

# set up loggers and checkpoints
log_dir = os.path.join(root_dir, "logs")
print("LOGDIR:", log_dir, os.path.exists(log_dir))
wandb_logger = WandbLogger(
    project="GR_AMOS"
)


# add multiple parameters
wandb_logger.experiment.config.update(hparams)

checkpoint_dir = os.path.join(root_dir,"checkpoints")
val_loss_checkpoint = ModelCheckpoint(save_top_k=3, monitor="val_loss",every_n_epochs=2, dirpath=checkpoint_dir, filename="AMOS_{epoch:02d}_{global_step}_{val_loss}")
val_dice_checkpoint = ModelCheckpoint(save_top_k=3, monitor="val_dice",every_n_epochs=2, dirpath=checkpoint_dir, filename="AMOS_{epoch:02d}_{global_step}_{val_dice}")

# initialise Lightning's trainer.

trainer = pytorch_lightning.Trainer(
    max_epochs=hparams["max_epochs"],
    logger=wandb_logger,
    enable_checkpointing=True,
    num_sanity_val_steps=1,
    log_every_n_steps=10,
    callbacks=[val_dice_checkpoint,val_loss_checkpoint]
)
trainer.fit(net)