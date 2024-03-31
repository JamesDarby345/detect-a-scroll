#Detect a scroll

This repo has the tools I used to create the masked data for Scrolls3 & 4. It isnt polished but works, below is the relevant files, model checkpoint and dataset used, afterwards I will explain why I did this instead of using the SAM implementation from april/may 2023.

## Brief notes on how to reproduce/use

Set up the detectron2 environment as per facebooks docs below in this readme, this can be annoying to do, good luck.

Download the finetuned checkpoint of the detectron2 mask_rcnn_R_101_FPN_3x for instance segmentation model, and the dataset with a coco annotations .json file from here: https://drive.google.com/drive/folders/1bQaFEFCtlnV70XyH2x_0euN3iOHSrbkP?usp=sharing

Put the .pth in a checkpoints folder, and the data + annotations in a test_data folder. Make the paths in train.ipynb in demo/ point to the checkpoint if you want to resume training & the data if you want to train the checkpoint.

Use the apply_masks.py file to load the checkpoint, do inference on the data and save the masked volume as a .tif, .jpg or just the unapplied mask. See the comments in the file for how to use in detail.

Note that this is just if you are interested in reproducing the result, the masked data will be /hopefully has been at time or reading, uploaded to dl.ash2text.org to download easily, as well as the ome-ZARR files.

## Why this repo instead of the old SAM one
In april/may 2023 I masked Scroll 1 & 2 data with SAM, the current SOTA general purpose segmentation model. But as with most large 2d vision models, it was trained on 'natural' images, ones you might take with a camera. With some supporting code, hacks and manual review of the masks I was able to use it to mask all the Scroll 1 and 2 volume data, but it took me a week to do it. I was heavily involved with the process. SAM is also a relatively heavy model, taking ~20-30 seconds for inference on a single .tif on the rtx2070S I was running it on. This time I knew I didnt have the time to spend a week babysitting SAM to mask the data and wanted it to run faster and without me in the loop nearly as much. 

Thus I choose to fork detectron2 as it provides an interface to try a bunch of models. Though none worked perfectly out of the box. I deicided to create a dataset of coco annotations of scroll instances with darwin v7 to finetune a model so it would work better. The goal was that it would not require me to diligently check each mask to make sure it hadnt deicided the empty space was the scroll like SAM had been doing on occasion. After a few iterations of the annotated dataset and multiple training runs I got a model that worked well. Now to ensure the masks were not incorrect I could just create an ome-ZARR from them and see if anything bizzare had happened by inspecting it in khartes, far faster than checking mask by mask. Additionally the finetuned mask_rcnn_R_101_FPN_3x model I choose could do inference on a .jpg version of the volume in ~1s on my M2 cpu (as mps is annoying when cuda is the default) instead of 20-30s. 

Additionally I stopped being stupid and converted all the volumes from .tif's to .jpg's at the same res but 90-95% smaller data size, which allowed me to annotate them far faster (as the .tif files lagged annotation software extremly hard) and also train the model and do inference much faster. The volume masking gains no advantage from the .tif files as whether or not a pixel is scroll is not dependant on some crazy high data precision. Once the binary mask is produced from the .jpg, it can be applied to the .tif files easily, exponentially increasing the speed of the process.

Essentially this new approach of converting the volumes to .jpg, creating the masks with a lightweight finetuned model and applying them to the .tif allows the masking computation to be done in a single day or less instead of babysat over a week. This assumes you have the storage space to have the full volume downloaded at once. Now the only bottleneck is in the upload & download. In theory the process could just be run on the server to get around that as well, though setting up detectron2, as with most 'fancy' models is a pain to do. Anouther approach could be converting volumes to .jpg on the server, downloading those much faster, creating unapplied masks as .png, uploading those and applying them with a script on the server without having to upload/download full .tif volumes (a multi-day undertaking). That way the masks could be inspected locally on the .jpg files as well if deemed necessary.

Though the finetuned model doesnt make any major mistakes, I have redone some masks (by generating them on different files as I only make a maske every ~10 volumes) just to clean up some places were non-scroll pieces were masked as scroll. I have seen no failure cases of large/noticeable amounts of scroll being masked as not scroll yet, unlike the SAM approach, though thats no gurantee it couldnt happen, espcially with new data that looks substantially different.



<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

<a href="https://opensource.facebook.com/support-ukraine">
  <img src="https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB" alt="Support Ukraine - Help Provide Humanitarian Aid to Ukraine." />
</a>

Detectron2 is Facebook AI Research's next generation library
that provides state-of-the-art detection and segmentation algorithms.
It is the successor of
[Detectron](https://github.com/facebookresearch/Detectron/)
and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).
It supports a number of computer vision research projects and production applications in Facebook.

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>
<br>

## Learn More about Detectron2

Explain Like I’m 5: Detectron2            |  Using Machine Learning with Detectron2
:-------------------------:|:-------------------------:
[![Explain Like I’m 5: Detectron2](https://img.youtube.com/vi/1oq1Ye7dFqc/0.jpg)](https://www.youtube.com/watch?v=1oq1Ye7dFqc)  |  [![Using Machine Learning with Detectron2](https://img.youtube.com/vi/eUSgtfK4ivk/0.jpg)](https://www.youtube.com/watch?v=eUSgtfK4ivk)

## What's New
* Includes new capabilities such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, PointRend,
  DeepLab, ViTDet, MViTv2 etc.
* Used as a library to support building [research projects](projects/) on top of it.
* Models can be exported to TorchScript format or Caffe2 format for deployment.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

## Installation

See [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

## Getting Started

See [Getting Started with Detectron2](https://detectron2.readthedocs.io/tutorials/getting_started.html),
and the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
to learn about basic usage.

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
