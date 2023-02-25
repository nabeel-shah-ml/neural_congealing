# Neural Congealing: Aligning Images to a Joint Semantic Atlas
## <a href="https://neural-congealing.github.io/" target="_blank">Project Page</a> | <a href="https://arxiv.org/abs/2302.03956" target="_blank">Paper</a>


This repo contains the official PyTorch implementation of the paper "Neural Congealing: Aligning Images to a Joint Semantic Atlas".



https://user-images.githubusercontent.com/71140564/221129493-c714da4c-17e0-4bdc-9b23-b1c2f301bf3b.mp4

<br>

[//]: # (### Abstract)
>We present Neural Congealing – a zero-shot self-supervised framework for detecting and jointly aligning semantically-common content across a given set of images. 
>Our approach harnesses the power of pre-trained DINO-ViT features to learn: (i) a joint semantic atlas – a 2D grid that captures the mode of DINO-ViT features in the input set, and (ii) dense mappings from the unified atlas to each of the input images. We derive a new robust self-supervised framework that optimizes the atlas representation and mappings per image set, requiring only a few real-world images as input without any additional input information (e.g., segmentation masks). Notably, we design our losses and training paradigm to account only for the shared content under severe variations in appearance, pose, background clutter or other distracting objects. We demonstrate results on a plethora of challenging image sets including sets of mixed domains (e.g., aligning images depicting sculpture and artwork of cats), sets depicting related yet different object categories (e.g., dogs and tigers), or domains for which large-scale training data is scarce (e.g., coffee mugs). We thoroughly evaluate our method and show that our test-time optimization approach performs favorably compared to a state-of-the-art method that requires extensive training on large-scale datasets.


## Getting Started
### Environment Setup

```
git clone https://github.com/dolev104/neural_congealing.git
cd neural_congealing
export PYTHONPATH="${PYTHONPATH}:${PWD}"

conda env create -f environment.yaml
conda activate neural-congealing
```

### Download Sample Image Sets
Download sample image sets and their trained models:
```
gdown https://drive.google.com/uc?id=17tOao9v8QfonwvjCXPihJN5FA55PD39U&export=download
unzip data.zip
```

It will create a folder `data`:
```
neural_congealing
├── ...
├── data
│   ├── <images set 1>  # name of image set
│   │     ├── images
│   │     ├── images_cosegmentation
│   │     ├── sample_edits
│   │     └── checkpoint_epoch_<num>.pt
│   ├── <images set 2>
│   │     ├── images
│   │     ├── images_cosegmentation
│   │     ├── sample_edits
│   │     └── checkpoint_epoch_<num>.pt
│   ├── ...
│   └── checkerboard_gray_256.png
└── ...
```
`images_cosegmentation` contains the output of `cosegmentation.py` (see below).

## Evaluation
Run the following for evaluating a trained model:
```
python evaluate_model.py --checkpoint_path data/image_set_name/checkpoint_epoch_<num>.pt
```
Outputs will be saved in `evaluation_output`.

## Editing

For applying an edit on a set, use the average image as a template (located in `atlas_vis/atlas_space_average_image.png` after running `evaluate_model.py`). It is recommended that the RGBA edit image will be of a larger resolution than the average image (e.g., 256x256) for obtaining higher edit quality.
```
python propagate_edit.py --checkpoint_path data/image_set_name/model_checkpoint_file.pt --label_path path/to/edit/image
```
Outputs will be saved in `visuals_and_edits`.
<br>Sample edits can be found in the downloaded sample sets under `data/<images set 1>/sample_edits`.


## Training

### Data Preparation
#### Prepare image set
Choose a small set of semantically-related images (e.g., all containing a butterfly). Note that our experiments include 5-25 images, most around 12.
<br>To prepare your data, run the following:
```
python prepare_images.py --folder_path path/to/images/folder --name new_image_set --method [crop | pad]
```
Specifying `--method crop` will apply center crop to all images; `pad` will border pad images to square. 
<br>Processed images are saved in `data/new_image_set/images`.

#### Inspect Set in DINO-ViT space
Our method aligns images in the DINO-ViT feature space; hence for the alignment to work well, the features should represent well the shared content, as well as the semantic association across the set.

For visualizing the similarities between image features, we take the interactive script from [dino-vit-features](https://github.com/ShirAmir/dino-vit-features/blob/main/inspect_similarity.py), and expand it to work on a set of images.
This visualizes the similarity of a chosen feature in the selected image (`ref_image_idx`) to all rest of the images' features.
```
python utils_atlas/inspect_similarity_set.py --folder_path data/new_image_set --ref_image_idx 0
```

#### Extract Initial Saliency Masks
We use the method of cosegmentation that was proposed by Amir et al. ([dino-vit-features](https://github.com/ShirAmir/dino-vit-features)). For extracting initial saliency masks, run the following:
```
python cosegmentation.py --folder_path data/new_image_set
```
Outputs are saved in `data/new_image_set/images_cosegmentation` (only the subfolder `mask` is used for training).<br>
You can find the original code [here](https://github.com/ShirAmir/dino-vit-features/blob/main/cosegmentation.py).

### Train
Update the `config.yaml`. The main things: update the path in `data_folder`, set `ext_horizontal_flips` to `True` if you wish to use horizontal flips during training, and set desired logging parameters. 
<br>Run the following command to start training:
```
python train.py --config configs/config.yaml
```
Intermediate results will be saved to `results` during training. The frequency of saving intermediate results/checkpoints is indicated by the `log_images_freq`/`save_model_freq` flag of the configuration, respectively.

## Coming soon
- [ ] More sample sets
- [ ] PCK-Transfer evaluation

## Acknowledgements
We thank [GANgealing](https://github.com/wpeebles/gangealing) and [dino-vit-features](https://github.com/ShirAmir/dino-vit-features) for their helpful code!

## Citation
If you found our code useful, please consider citing:
```
@article{ofriamar2023neural,
         title      = {Neural Congealing: Aligning Images to a Joint Semantic Atlas},
         author     = {Ofri-Amar, Dolev and Geyer, Michal and Kasten, Yoni and Dekel, Tali},
         journal    = {arXiv preprint arXiv:2302.03956},
         year       = {2023}
}
```
