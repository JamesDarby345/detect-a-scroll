import torch, detectron2 

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

import cv2, tifffile
import numpy as np
import matplotlib.pyplot as plt

from detectron2.engine import DefaultTrainer
import time

register_coco_instances("my_dataset_train", {}, "../test_data/finetune_detect-a-scroll/train/train.json", "../test_data/finetune_detect-a-scroll/train")
register_coco_instances("my_dataset_val", {}, "../test_data/finetune_detect-a-scroll/val/val.json", "../test_data/finetune_detect-a-scroll/val")

cfg = get_cfg()
cfg.OUTPUT_DIR = "../checkpoints"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 24  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 1000 iterations seems good enough for this dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # Default is 512, using 256 for this dataset.
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # We have 4 classes.
# NOTE: this config means the number of classes, without the background. Do not use num_classes+1 here.
# Set the checkpoint period
cfg.SOLVER.CHECKPOINT_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) #Create an instance of of DefaultTrainer with the given congiguration
trainer.resume_or_load(resume=True) #Load a pretrained model if available (resume training) or start training from scratch if no pretrained model is available

#boilerplate that could be pruned, but works as is
#---------------------------------------------------
#useful code to change -> see make_masks.ipynb to see visulizations of the masks

def write_image_tifffile(file_path, image):
    try:
        tifffile.imwrite(file_path, image, compression="lzw")
    except Exception as e:
        print(e)
        return False
    return True

def dilate_mask(mask, dilation_percentage):
    kernel_size = int(max(mask.shape) * dilation_percentage / 100)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask

def apply_mask(image, mask):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image



#CHANGE these variables
input_folder="/Users/jamesdarby/Documents/VesuviusScroll/GP/Vesuvius_Data_Download/Scroll4_PHerc_1667/PHerc1667.volpkg/volumes/20231107190228/"
range_of_images = range(0, 501)
# range_of_images = range(10000, 20000)
final_folder = "masked_volumes"
output_folder = f"/Users/jamesdarby/Documents/VesuviusScroll/GP/Vesuvius_Data_Download/Scroll4_PHerc_1667/{final_folder}/"
tif_name_length = 5
#better for end of scrolls
model_checkpoint = "larger_instance_run/model_ends_of_scroll.pth" 
#better for middle of scrolls
# model_checkpoint = "model_0000999.pth" 
dilation_percentage = 4
mask_recalculation_interval = 10

os.makedirs(output_folder, exist_ok=True)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_checkpoint)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

count = 0
for i in range_of_images:
    file_name = str(i).zfill(tif_name_length) + ".tif"
    file_path = input_folder + file_name

    #tifffile keeps the tif in its original uint16 format, cv2 reads it as a uint8
    # the model cant take uint16, so we provide the cv2 image to the model
    # we then use the tifffile image to apply the mask and save the result
    # so the resulting image doesnt lose the uint16 percision
    try:
        img = tifffile.imread(file_path) #to apply mask to and save
        if count % mask_recalculation_interval == 0:
            cvimg = cv2.imread(file_path)
    except Exception as e:
        print(e)
        # count += 1
        continue

    if count % mask_recalculation_interval == 0:
        outputs = predictor(cvimg)
        masks = outputs["instances"].pred_masks.cpu().numpy()
        combined_mask = np.logical_or.reduce(masks).astype(np.uint8)
        dilated_mask = dilate_mask(combined_mask, dilation_percentage)

    masked_image = apply_mask(img, dilated_mask)

    write_image_tifffile(output_folder + file_name, masked_image)
    print("masked ", file_name)
    count += 1
