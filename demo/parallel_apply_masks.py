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
from itertools import chain
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image(cur_range):
    file_path = os.path.join(input_folder, str(cur_range[0]).zfill(tif_name_length) + ".tif")
    cvimg = cv2.imread(file_path)
    outputs = predictor(cvimg)
    masks = outputs["instances"].pred_masks.cpu().numpy()
    combined_mask = np.logical_or.reduce(masks).astype(np.uint8)
    final_mask = dilate_mask(combined_mask, dilation_percentage)
    print(f"Processing range {cur_range}")
    for i in cur_range:
        img = cv2.imread(os.path.join(input_folder, str(i).zfill(tif_name_length) + ".tif"))
        if save_masks:
            final_mask_to_save = final_mask * 255
            write_image_cv2(output_folder_masks + "/" + str(i).zfill(tif_name_length) + ".png", final_mask_to_save)
        
        if apply_masks_tif or apply_masks_jpg:
            masked_image = apply_mask(img, final_mask)
        if apply_masks_tif:
            file_name = str(i).zfill(tif_name_length) + ".tif"
            write_image_tifffile(output_folder + "/" + file_name, masked_image)
        if apply_masks_jpg:
            file_name = str(i).zfill(tif_name_length) + ".jpg"
            write_image_cv2(output_folder + "_jpg" + "/" + file_name, masked_image)
        # print(f"Processed image {i}, {input_folder} {str(i).zfill(tif_name_length)} .tif")
    return True

def parallel_process(range_of_images, mask_recalculation_interval):
    range_list = []
    for i in range(0, len(range_of_images), mask_recalculation_interval):
        cur_range = range_of_images[i:i+mask_recalculation_interval]
        range_list.append(cur_range)
    with ThreadPoolExecutor() as executor:  # Adjust max_workers as needed
        
        futures = [executor.submit(process_image, range) for range in range_list]

        # Wait for all tasks to complete and handle results
        for future in as_completed(futures):
            result = future.result()
                
        

if __name__ == "__main__":

    register_coco_instances("my_dataset_train", {}, "../test_data/output.json", "../test_data/")

    cfg = get_cfg()
    cfg.OUTPUT_DIR = "../checkpoints"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
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

    def write_image_cv2(file_path, image):
        try:
            if image.dtype == np.uint16:
                image = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
            cv2.imwrite(file_path, image) #defaults jpg quality to 95
        except Exception as e:
            print(e)
            return False
        return True

    def dilate_mask(mask, dilation_percentage):
        kernel_size = int(max(mask.shape) * dilation_percentage / 100)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        return dilated_mask

    def erode_mask(mask, erosion_percentage):
        kernel_size = int(max(mask.shape) * erosion_percentage / 100)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        return eroded_mask

    def apply_mask(image, mask):
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image

    """
    How to use once environment is setup
    point input and output folder to the correct locations of the .tif files
    set tif_name_length to the number of digits in the tif file names
    set model_checkpoint to the name of the model checkpoint to use
    The ends of scrolls model is better for the ends of scrolls, 
    the other model is better for the middle of scrolls
    set dilation_percentage to 1.5 for the ends of scrolls, and 4+ for the middle of scrolls
    set the range of images to the range of images you want to mask

    the mask_recalculation_interval tells the script how many volumes to skip before recalculating the mask
    this helps speed up runs, but if too large, the mask will be out of date

    For a first pass, sset the mask recalculation interval as low as possible, recgonizing each mask calculation
    takes ~1-3 seconds. Then inspect the results and find areas where the masking is suboptimal. Find the closest
    larger/better mask and set the start of range of images to the first image with that better mask, then set the
    end to the next image with a better mask and set the mask recalculation interval to that number. Then run the script.
    This will use that good mask on all the images in that range. Then repeat this process until the end of the scroll.

    """

    #CHANGE these variables

    #Scroll 1
    # input_folder = "/Volumes/16TB_RAID_0/Scroll1/Scroll1.volpkg/volumes/20230205180739" 
    # output_folder = "/Volumes/16TB_RAID_0/Scroll1/masked_volumes"
    # output_folder_masks = "/Volumes/16TB_RAID_0/Scroll1/volume_masks"

    #Scroll 3
    # input_folder = "/Volumes/16TB_slow_RAID_0/Scroll3/PHerc0332.volpkg/volumes/20231027191953" 
    # output_folder = f"/Volumes/16TB_slow_RAID_0/Scroll3/masked_volumes"
    # output_folder_masks = f"/Volumes/16TB_slow_RAID_0/Scroll3/volume_masks"

    #Scroll 4
    input_folder = "/Volumes/16TB_RAID_0/Scroll4/PHerc1667.volpkg/volumes/20231107190228"
    output_folder = "/Volumes/16TB_slow_RAID_0/Scroll4/masked_volumes"
    output_folder_masks = "/Volumes/16TB_slow_RAID_0/Scroll4/volume_masks"

    tif_name_length = 5

    # range_of_images = range(0, 501)
    # range_of_images = range(13150, 14376)
    range_of_images = range(13640, 13806)

    # range_of_images = chain(range(0, 800), range(22000, 22941))

    range_of_images = range(13600, 13670)
    range_of_images = range(0, 27000)

    model_checkpoint = "model_0050999.pth"

    dilation_percentage = 3 #~1.5 for end of scrolls, 4+ for middle of scrolls
    mask_recalculation_interval = 10 #how many volumes to skip before recalculating the mask
    save_masks = True #if true, will save the masks as .png files
    apply_masks_tif = True #if true, will apply the masks to the images and save the results as .tif files
    apply_masks_jpg = True #if true, will apply the masks to the images and save the results as .jpg files

    manual_erode = False #if true, will erode the mask by the amount in erode_amount every mask recalculation interval instead of predicting with model.
    erode_amount = 0.1 #percentage to erode the mask by

    if apply_masks_tif:
        os.makedirs(output_folder, exist_ok=True)
    if apply_masks_jpg:
        os.makedirs(output_folder+"_jpg", exist_ok=True)
    if save_masks:
        os.makedirs(output_folder_masks, exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_checkpoint)  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    final_mask = None

    start_time = time.time()
    parallel_process(range_of_images, mask_recalculation_interval)
    print(f"Total time: {time.time() - start_time}")
