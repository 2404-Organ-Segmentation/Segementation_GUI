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

# TODO: Add transforms that were used for swinunetr model validation
SwinUNETR_transforms = Compose(
    [

    ]
)


class ParentModel:

    def __init__(self, model, transforms, model_path: str, data_folder: str, debug: bool = False) -> None:
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
        self.transforms = transforms

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

    def inference(self, output_folder) -> None:
        """! Runs the prediction on the files located under self.data_folder, will save the files as Nifti (.nii.gz)
        format under output_folder. If output_folder is not specified, then it will be saved to the folder where the
        data was originally taken from.

        @:param output_folder: The folder path to save the nifti images as a string. If None, then it will save to the
        folder where the data files are located. (self.data_folder)
        @:return None
        """

        counter = 0
        with torch.no_grad():
            for i, test_data in enumerate(self.val_loader):
                # Make prediction
                img = test_data["image"].to(self.device)
                test_data["pred"] = sliding_window_inference(img, (96, 96, 96), 4, self.model, overlap=0.8)

                # Post-processing transforms
                # Source: https://github.com/MASILab/3DUX-Net/tree/14ea46b7b4c4980b46aba066aaaa24b1d9c1bb0d
                post_transforms = Compose([
                    EnsureTyped(keys="pred"),
                    Activationsd(keys="pred", softmax=True),
                    Invertd(
                        keys="pred",  # invert the `pred` data field, also support multiple fields
                        transform=UNETR_transforms,
                        orig_keys="image",
                        # get the previously applied pre_transforms information on the `img` data field,
                        # then invert `pred` based on this information. we can use same info
                        # for multiple fields, also support different orig_keys for different fields
                        meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
                        orig_meta_keys="image_meta_dict",
                        # get the meta data from `img_meta_dict` field when inverting,
                        # for example, may need the `affine` to invert `Spacingd` transform,
                        # multiple fields can use the same meta data to invert
                        meta_key_postfix="meta_dict",
                        # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
                        # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
                        # otherwise, no need this arg during inverting
                        nearest_interp=False,
                        # don't change the interpolation mode to "nearest" when inverting transforms
                        # to ensure a smooth output, then execute `AsDiscreted` transform
                        to_tensor=True,  # convert to PyTorch Tensor after inverting
                    ),
                    AsDiscreted(keys="pred", argmax=True),
                    KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 3]),
                    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_folder,
                               output_postfix="temp", output_ext=".nii.gz", resample=True, separate_folder=False),
                ])
                test_data = [post_transforms(j) for j in decollate_batch(test_data)]

                # Small modification to affine matrix
                self.__load_and_translate(output_folder=output_folder, file_name=self.files[counter])
                counter += 1

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

    def __load_and_translate(self, output_folder, file_name) -> None:
        """! Helper function that loads the saved file from monai and applies the necessary affine matrix modifications
        to it, then deletes the temporary monai file and saves as the proper nifti file.

        @:param output_folder: The path to the folder that contains the temporary monai saved file.
        @:param file_name: The name of the file that was being analyzed.
        @:return None
        """
        temp_name = re.sub(r"\.nii\.gz$", "_temp.nii.gz", file_name)
        temp_file_path = os.path.join(output_folder, temp_name)
        seg_img = nib.load(temp_file_path)
        self.__debug(f"segm affine is {seg_img.affine}")
        self.__debug(f"segm shape is {seg_img.shape}")

        new_affine = seg_img.affine
        new_affine[:3, 3] = [0, 0, 0]
        new_affine[1, 1] = -1 * new_affine[1, 1]
        self.__debug(f"New affine is {new_affine}")

        new_name = re.sub(r"\.nii\.gz$", "-segmented.nii.gz", file_name)

        nib.save(
            nib.Nifti1Image(seg_img.get_fdata(), affine=new_affine),
            os.path.join(output_folder, new_name)
        )

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            self.__debug(f"File '{temp_file_path}' deleted successfully")
        else:
            self.__debug(f"File '{temp_file_path}' does not exist")


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
        super().__init__(model=unetr_model, transforms=UNETR_transforms, model_path=model_path,
                         data_folder=data_folder, debug=debug)


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
        super().__init__(model=swin_unetr_model, transforms=SwinUNETR_transforms, model_path=model_path,
                         data_folder=data_folder, debug=debug)


if __name__ == "__main__":
    # Only run this file directly for debugging
    trainer = UnetRModel(model_path="./best_metric_model_3dUNETR52200.pth",
                         data_folder="./testing", debug=True)
    trainer.inference(output_folder="./testing")
