import sys
import os
import shutil
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import QCoreApplication, Qt

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
        self.pushButton_download.clicked.connect(self.select_output_location)

        self.textEdit_upload.setReadOnly(True)
        self.textEdit_run.setReadOnly(True)

        self.clear_upload_folder()

    def clear_upload_folder(self):
        # Clear the upload folder
        for filename in os.listdir(os.path.join(self.base_path, "uploads")):
            file_path = os.path.join(os.path.join(self.base_path, "uploads"), filename)
            os.unlink(file_path)

    def upload_files(self):
        # Slot for handling file upload
        filenames, _ = QFileDialog.getOpenFileNames(self, "Upload Files")
        for filename in filenames:
            self.textEdit_upload.append(filename + " uploaded.")

            destination = os.path.join(os.path.join(self.base_path, "uploads"), os.path.basename(filename))
            shutil.copyfile(filename, destination)

    def select_model(self, index):
        # Slot for handling model selection
        model_name = self.comboBox_model.itemText(index)
        print("Selected model:", model_name)

    def run_segmentation(self):
        # Slot for running the segmentation process
        print("Running segmentation...")

    def select_output_location(self):
        # Slot for downloading files
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Location")
        if folder_path:
            self.output_folder_path = folder_path
            self.textEdit_run.append("Output folder selected: " + self.output_folder_path)

    
def main():
    app = QApplication(sys.argv)

    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    window = SegmentationApp()
    window.show()
    
    sys.exit(app.exec_())