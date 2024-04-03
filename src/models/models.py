# =========================================================================== #
#
#   Title: models.py
#   Date: 2024-04-02
#
#   Description: Classes to setup and predict using UNETR and SWINUNETR. Reads
#                files from a folder and uses pretrained models.
#
# =========================================================================== #
import os
import re
import nibabel as nib
import numpy as np
import torch
import scipy.ndimage as ndimage
from monai import data
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR, UNETR

UNETR_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 2.0),
            mode="bilinear",
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
    ]
)
# TODO: Add transforms that were used for swinunetr model training


class ParentModel:

    def __init__(self, model, model_path: str, data_folder: str, debug: bool = False) -> None:
        """! Parent constructor for model prediction. Defines the model type that is used as well as the paths for
        loading the pretrained model, loading and saving the data
        @:param model: The model that is going to be used for predictions. Should be monai UNETR or SwinUNETR.
        @:param model_path: The path to the pretrained model as a string. Should include the model .pth file.
        @:param data_folder: The folder where the data is located as string. All files in this folder should be medical
        images.
        @:param debug: Boolean that enables debug messages. Defaults to false to disable messages.
        @:return None
        """
        self.val_loader = None
        self.debug_mode = debug

        pretrained_pth = model_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.file_dicts = []
        self.files = []
        self.data_folder = data_folder
        self.load_dataset()

        self.model = model
        self.model.to(device)
        self.model.load_state_dict(torch.load(pretrained_pth))
        self.model.eval()

    def test_output(self, output_folder: str = None) -> None:
        """! Runs the prediction on the files located under self.data_folder, will save the files as Nifti (.nii.gz)
        format under output_folder. If output_folder is not specified, then it will be saved to the folder where the
        data was originally taken from.
        @:param output_path: The folder path to save the nifti images as a string. If None, then it will save to the
        folder where the data files are located. (self.data_folder)
        @:return None
        """

        counter = 0
        if output_folder is None:
            output_folder = self.data_folder

        with torch.no_grad():
            for i in self.val_loader:
                # Obtain the name for the file and append -segmented to it
                file_name = re.sub(r"\.nii\.gz$", "-segmented.nii.gz", self.files[counter])

                # Need information about the original image to find a location in space
                img = i["image"][0]
                original_image = nib.load(self.file_dicts[0]["image"])
                header = original_image.header

                val_inputs = torch.unsqueeze(img, 1).cuda()
                _, _, h, w, d = val_inputs.shape
                target_shape = (h, w, d)
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, self.model, overlap=0.8)
                original_affine = val_outputs.affine
                original_shape = val_outputs.shape

                # Sets the size of voxels and location with respect to origin
                # TODO: Make the offset adjustable as a parameter since I cannot figure out how to centre the image
                offset = (original_image.shape[1] - original_shape[3]) * original_image.affine[1, 1]
                original_affine[:3, 3] = torch.tensor([original_shape[2] / -1 * original_affine[0, 0], original_shape[3] / -1 * original_affine[1, 1] + offset, 0], dtype=torch.float64)
                original_affine[1, 1] = -1 * original_affine[1, 1]

                # Post-processing + saving
                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
                val_outputs = self.resample_3d(val_outputs, target_shape)
                nib.save(
                    nib.Nifti1Image(val_outputs, affine=original_affine, header=header),
                    os.path.join(output_folder, file_name)
                )

                counter += 1
                self.debug(f"offset is {offset}")
                self.debug(f"affine is {original_affine}")
                self.debug(f"shape is {original_shape}")

    def load_dataset(self) -> None:
        """! Loads and preprocesses the data specified in under self.data_folder. Will save the data as a Monai
        Dataloader and apply the relevant transforms that were used for training.
        @:return None
        """

        self.load_files_from_folder()
        test_dataset = data.Dataset(data=self.file_dicts, transform=UNETR_transforms)
        self.val_loader = data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def load_files_from_folder(self) -> None:
        """! Loads the files into a list of dictionaries to be read by Monai's built in dataset. This needs to be
        formatted in this specific way so the transforms can be properly applied (the transforms are expecting specific
        keys). The files that are loaded are all the files in the folder specified by self.data_folder. This is a mock
        of Monai's load_decathlon_datalist().
        @:return None
        """
        self.file_dicts.clear()
        self.files.clear()

        for root, dirs, files in os.walk(self.data_folder):
            for file in files:
                file_path = os.path.join(root, file)
                image_dict = {
                    "image": file_path,
                }
                self.files.append(file)
                self.file_dicts.append(image_dict)

    def debug(self, message: str) -> None:
        """! Debug print statements, allows debug messages to be sent if self.debug_mode is True.
        @:param message: The message that is sent
        @:return None
        """
        if self.debug_mode:
            print(message)
        return None

    @staticmethod
    def resample_3d(img, target_size):
        """! Helper function that resizes the voxels for the output segmentation image. This will zoom the image to be the
        same size as the input dimensions.

        @:param img: The image that is being resized, should be a 3 dimensional image.
        @:param target_size: The size that the image needs to be.

        @:return img_resampled: The image that has been zoomed to the appropriate size.
        """
        imx, imy, imz = img.shape
        tx, ty, tz = target_size
        zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
        img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
        return img_resampled


class UnetRModel(ParentModel):

    def __init__(self, model_path: str, data_folder: str, debug: bool = False, modality: int = 1,
                 output_channels: int = 14) -> None:
        """! Child constructor for model prediction. Defines the model type that is used as well as the paths for
        loading the pretrained model, loading and saving the data.

        @:param model: The model that is going to be used for predictions. Should be monai UNETR or SwinUNETR.
        @:param model_path: The path to the pretrained model as a string. Should include the model .pth file.
        @:param data_folder: The folder where the data is located as string. All files in this folder should be medical
        images.
        @:param debug: Boolean that enables debug messages. Defaults to false to disable messages.
        @:param modality: Number of input dimensions / labels as an int.
        @:param output_channels: Number of output labels as an int.
        @:return None
        """
        unetr_model = UNETR(
            in_channels=modality,
            out_channels=output_channels,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
        super().__init__(model=unetr_model, model_path=model_path, data_folder=data_folder, debug=debug)


class SwinUnetRModel(ParentModel):

    def __init__(self, model_path: str, data_folder: str, debug: bool = False, modality: int = 1,
                 output_channels: int = 14) -> None:
        """! Child constructor for model prediction. Defines the model type that is used as well as the paths for
        loading the pretrained model, loading and saving the data.

        @:param model: The model that is going to be used for predictions. Should be monai UNETR or SwinUNETR.
        @:param model_path: The path to the pretrained model as a string. Should include the model .pth file.
        @:param data_folder: The folder where the data is located as string. All files in this folder should be medical
        images.
        @:param debug: Boolean that enables debug messages. Defaults to false to disable messages.
        @:param modality: Number of input dimensions / labels as an int.
        @:param output_channels: Number of output labels as an int.
        @:return None
        """
        swin_unetr_model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=modality,
            out_channels=output_channels,
            feature_size=48,
            use_checkpoint=True,
        )
        super().__init__(model=swin_unetr_model, model_path=model_path, data_folder=data_folder, debug=debug)


if __name__ == "__main__":
    # Only run this file directly for debugging
    trainer = UnetRModel(model_path="./best_metric_model_3dUNETR52200.pth",
                         data_folder="./testing", debug=True)
    trainer.test_output(output_folder="./testing")
