import sys
import os
import shutil
import requests
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import QCoreApplication, Qt

from .models import UnetRModel, SwinUnetRModel

host = "http://127.0.0.1:5000"

class SegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Segmentation App")

        self.base_path = os.path.dirname(os.path.realpath(__file__))

        ui_path = os.path.join(self.base_path, "ui/mainwindow.ui")
        loadUi(ui_path, self)

        self.pushButton_upload.clicked.connect(self.upload_files)
        self.comboBox_model.currentIndexChanged.connect(self.select_model)
        self.pushButton_run.clicked.connect(self.run_segmentation)
        self.pushButton_download.clicked.connect(self.download_files)

        self.textEdit_upload.setReadOnly(True)
        self.textEdit_run.setReadOnly(True)

        self.model = None
        self.training = False

    def upload_files(self):
        # Slot for handling file upload
        filenames, _ = QFileDialog.getOpenFileNames(self, "Upload Files")
        for filename in filenames:
            if filename.lower().endswith(".nii.gz"):
                with open(filename, 'rb') as f:
                    response = requests.post(f"{host}/upload", files={'file': f})
                
                self.textEdit_upload.append(response.text)
            else:
                self.textEdit_upload.append(f"{filename} has an invalid file format. Please upload .nii.gz files only.")


    def select_model(self, index):
        # Slot for handling model selection
        model_name = self.comboBox_model.itemText(index)
        if model_name == "UNETR" or model_name == "SWIN-UNETR":
            self.textEdit_run.append("Model selected: " + model_name)
            self.model = model_name

    def run_segmentation(self):
        # Check that files have been uploaded, and a model has been selected
        if self.model is None:
            self.textEdit_run.append("Please select a model before running segmentation.")
            return
        if not self.training:
            self.training = True
            # Slot for running the segmentation process
            self.textEdit_run.append("Running segmentation...")

            model_id = 0 if self.model == "UNETR" else 1
            response = requests.post(f"{host}/run", data={'model_id': model_id})

            self.textEdit_run.append(response.text)
            self.training = False

    def download_files(self):
        # Slot for downloading files
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Location")
        if folder_path:
            self.output_folder_path = folder_path
            self.textEdit_run.append("Output folder selected: " + self.output_folder_path)
            
        response = requests.get(f"{host}/download")

        if response.status_code == 200:
            # Create the destination folder if it doesn't exist
            os.makedirs(self.output_folder_path, exist_ok=True)

            # Write the downloaded zip file to disk
            zip_file_path = os.path.join(self.output_folder_path, 'output_files.zip')
            with open(zip_file_path, 'wb') as f:
                f.write(response.content)

            # # Extract the downloaded zip file if needed
            # shutil.unpack_archive(zip_file_path, self.output_folder_path)
            # os.remove(zip_file_path)

            print(f"Files downloaded and saved to {self.output_folder_path}")
        else:
            print(f"Error: {response.text}")

    
def main():
    app = QApplication(sys.argv)

    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    window = SegmentationApp()
    window.show()
    
    sys.exit(app.exec_())