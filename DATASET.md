# Multimodal Datasets Documentation

This document outlines the datasets used, including their sources, content descriptions, feature extraction methods, and data split statistics.



### MELD (Multimodal EmotionLines Dataset)
* **Data Source and Content:** A large-scale dataset for emotion recognition in conversations. It contains over 13,000 utterances from the TV show *Friends*, with audio, video, and text.
* **Link:** [GitHub Repository](https://github.com/declare-lab/MELD)
* **Feature Extraction:** (Not specified in source)
* **Data Splits:** The official split is used, containing **9,989** training, **1,109** validation, and **2,610** test utterances.

### IEMOCAP
* **Data Source and Content:** The Interactive Emotional Dyadic Motion Capture (IEMOCAP) database is a popular dataset for emotion recognition, containing approximately 12 hours of audiovisual data from ten actors.
* **Link:** [Official Site](https://sail.usc.edu/iemocap/)
* **Feature Extraction:** We follow [this repository](https://github.com/soujanyaporia/multimodal-sentiment-analysis/tree/master?tab=readme-ov-file) to preprocess the features.
* **Data Splits:** The official split is used: **5,810** training and **1,623** test utterances.
    * *Note:* Since the official split does not include a dedicated validation set, we reuse the test set as the validation set.

### CH-SIMS
* **Data Source and Content:** A fine-grained single- and multi-modal sentiment analysis dataset in Chinese. It contains over 2,200 short videos with unimodal and multimodal annotations.
* **Link:** [GitHub Repository](https://github.com/thuiar/MMSA)
* **Feature Extraction:** We use the pre-extracted features provided by the dataset without any additional modifications.
* **Data Splits:** The official split is used: **1,368** training, **456** validation, and **457** test utterances.

### CH-SIMSv2
* **Data Source and Content:** An extended version of CH-SIMS containing over 2,200 short videos with unimodal and multimodal annotations.
* **Link:** [Project Page](https://thuiar.github.io/sims.github.io/chsims)
* **Feature Extraction:** We use the pre-extracted features provided by the dataset without any additional modifications.
* **Data Splits:** The official split is used: **2,722** training, **647** validation, and **1,034** test utterances.

### MVSA-Single
* **Data Source and Content:** A Multi-View Sentiment Analysis dataset containing image-text posts from Twitter. The "Single" variant contains posts where the sentiment label is consistent across annotators.
* **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/vincemarcs/mvsasingle)
* **Feature Extraction:** We follow [MMBT](https://github.com/facebookresearch/mmbt) to prepare the splits. Images are encoded using `ResNet`, and tweet text is encoded using `BERT`.
* **Data Splits:** The official split is used: **1,555** training, **518** validation, and **519** test utterances.

### Twitter2015
* **Data Source and Content:** Originating from SemEval tasks, these datasets were extended for multimodal aspect-based sentiment analysis, pairing tweets with relevant images.
* **Link:** [Archive](https://archive.org/details/twitterstream)
* **Feature Extraction:** A `ResNet` is used to encode the image component, and a `BERT` model is used to encode the text.
* **Data Splits:** The official split is used: **3,179** training, **1,122** validation, and **1,037** test utterances.

### Twitter1517
* **Data Source and Content:** Extended dataset for multimodal aspect-based sentiment analysis.
* **Link:** [GitHub Repository](https://github.com/code-chendl/HFIR)
* **Feature Extraction:** A `ResNet` is used to encode the image component, and a `BERT` model is used to encode the text.
* **Data Splits:** We adopt a 7:1:2 split (Seed: 42): **3,270** training, **467** validation, and **935** test utterances.

### MAMI
* **Data Source and Content:** A dataset for Misogynistic Meme Detection, containing over 10,000 image-text memes annotated for misogyny and other categories.
* **Link:** [GitHub Repository](https://github.com/MIND-Lab/SemEval2022-Task-5-Multimedia-Automatic-Misogyny-Identification-MAMI-)
* **Feature Extraction:** A `ResNet` is used to encode the image component, and a `BERT` model is used to encode the text.
* **Data Splits:** The official split is used: **9,000** training, **1,000** validation, and **1,000** test utterances.

### Memotion
* **Data Source and Content:** A dataset for analyzing emotions in memes. It contains 10,000 memes annotated for sentiment and three types of emotions (humor, sarcasm, motivation).
* **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k)
* **Feature Extraction:** A `ResNet` is used to encode the image component, and a `BERT` model is used to encode the text.
* **Data Splits:** We adopt an 8:1:1 split (Seed: 42): **5,465** training, **683** validation, and **683** test utterances.

### MUTE
* **Data Source and Content:** Targets troll-like behavior in image/text posts to detect harmful content.
* **Link:** [GitHub Repository](https://github.com/eftekhar-hossain/MUTE-AACL22)
* **Feature Extraction:** A `ResNet` is used to encode the image component, and a `BERT` model is used to encode the text.
* **Data Splits:** The official split is used: **3,365** training, **375** validation, and **416** test utterances.

### MultiOFF
* **Data Source and Content:** Focuses on identifying offensive content and its target in image/text posts.
* **Link:** [GitHub Repository](https://github.com/bharathichezhiyan/Multimodal-Meme-Classification-Identifying-Offensive-Content-in-Image-and-Text)
* **Feature Extraction:** A `ResNet` is used to encode the image component, and a `BERT` model is used to encode the text.
* **Data Splits:** The official split is used: **445** training, **149** validation, and **149** test utterances.

### MET-Meme (Chinese)
* **Data Source and Content:** A dataset for multimodal metaphor detection in memes (Chinese version).
* **Link:** [GitHub Repository](https://github.com/liaolianfoka/MET-Meme-A-Multi-modal-Meme-Dataset-Rich-in-Metaphors)
* **Feature Extraction:** A `ResNet` is used to encode the image component, and a `BERT` model is used to encode the text.
* **Data Splits:** We adopt a 7:1:2 split (Seed: 42): **1,609** training, **229** validation, and **461** test utterances.

### MET-Meme (English)
* **Data Source and Content:** A dataset for multimodal metaphor detection in memes (English version).
* **Link:** [GitHub Repository](https://github.com/liaolianfoka/MET-Meme-A-Multi-modal-Meme-Dataset-Rich-in-Metaphors)
* **Feature Extraction:** A `ResNet` is used to encode the image component, and a `BERT` model is used to encode the text.
* **Data Splits:** We adopt a 7:1:2 split (Seed: 42): **737** training, **105** validation, and **211** test utterances.

### Houston2013
* **Data Source and Content:** Provides HSI and LiDAR data over the University of Houston campus, covering 15 land use classes (IEEE GRSS Data Fusion Contest).
* **Link:** [Project Page](https://machinelearning.ee.uh.edu/?page_id=459)
* **Feature Extraction:** Following [this repo](https://github.com/songyz2019/rs-fusion-datasets). HSI data is processed by the `conv_hsi` encoder; LiDAR data is processed by the `conv_dsm` encoder.
* **Data Splits:** Official split: **2,817** training and **12,182** test.
    * *Note:* Test set is reused as validation set.

### Houston2018
* **Data Source and Content:** A more complex dataset covering 20 urban land use classes (IEEE GRSS Data Fusion Contest).
* **Link:** [Project Page](https://machinelearning.ee.uh.edu/2018-ieee-grss-data-fusion-challenge-fusion-of-multispectral-lidar-and-hyperspectral-data/)
* **Feature Extraction:** Following [this repo](https://github.com/songyz2019/rs-fusion-datasets). HSI data processed by `conv_hsi`; LiDAR processed by `conv_dsm`.
* **Data Splits:** Official split: **18,750** training and **2,000,160** test.
    * *Note:* Test set is reused as validation set.

### MUUFL Gulfport
* **Data Source and Content:** HSI and LiDAR data collected over the University of Southern Mississippi, Gulfport campus (11 urban land use classes).
* **Link:** [GitHub Repository](https://github.com/GatorSense/MUUFLGulfport)
* **Feature Extraction:** Following [this repo](https://github.com/songyz2019/rs-fusion-datasets). HSI data processed by `conv_hsi`; LiDAR processed by `conv_dsm`.
* **Data Splits:** Official split: **1,100** training and **52,587** test.
    * *Note:* Test set is reused as validation set.

### Trento
* **Data Source and Content:** Covers a rural area south of Trento, Italy. Combines HSI with LiDAR-derived DSM (6 classes).
* **Link:** [GitHub Repository](https://github.com/tyust-dayu/Trento/tree/b4afc449ce5d6936ddc04fe267d86f9f35536afd)
* **Feature Extraction:** Following [this repo](https://github.com/songyz2019/rs-fusion-datasets). HSI processed by `conv_hsi`; LiDAR processed by `conv_dsm`.
* **Data Splits:** Official split: **600** training and **29,614** test.
    * *Note:* Test set is reused as validation set.

### Berlin
* **Data Source and Content:** Co-registered HSI and SAR data for Berlin, Germany (8 urban land cover classes).
* **Link:** [Dataset Source](https://gfzpublic.gfz-potsdam.de/pubman/faces/ViewItemFullPage.jsp?itemId=item_1480927_5)
* **Feature Extraction:** Following [this repo](https://github.com/songyz2019/rs-fusion-datasets). HSI processed by `conv_hsi`. SAR data is processed by a `ResNet` encoder.
* **Data Splits:** Official split: **2,820** training and **461,851** test.
    * *Note:* Test set is reused as validation set.

### MDAS (Augsburg)
* **Data Source and Content:** Multi-sensor data for urban area classification in Augsburg, Germany, featuring HSI and SAR imagery.
* **Link:** [GitHub Repository](https://github.com/songyz2019/rs-fusion-datasets)
* **Feature Extraction:** Following [this repo](https://github.com/songyz2019/rs-fusion-datasets). HSI uses `conv_hsi`; SAR data uses `ResNet`.
* **Data Splits:** Official split: **761** training and **77,533** test.
    * *Note:* Test set is reused as validation set.

### ForestNet
* **Data Source and Content:** A dataset for wildfire prevention using Sentinel-2 imagery, topography (DSM), and weather data.
* **Link:** [GitHub Repository](https://github.com/spott/ForestNet)
* **Feature Extraction:** `ResNet` for satellite imagery, `ResNet` for topography, MLP for tabular weather data.
* **Data Splits:** Official split: **1,616** training, **473** validation, and **668** test.

### TCGA-BRCA
* **Data Source and Content:** TCGA Breast Cancer (BRCA) multi-omics dataset (gene expression, DNA methylation, copy number variation) for survival prediction.
* **Link:** [GitHub Repository](https://github.com/txWang/MOGONET)
* **Feature Extraction:** Each tabular omics modality is encoded via an independent fully-connected linear layer.
* **Data Splits:** Extracted sequentially: **657** training (75%), **43** validation (5%), and **175** test (20%).

### TCGA
* **Data Source and Content:** General TCGA multi-omics dataset for survival prediction (dataset selection via IntegrAO).
* **Link:** [NCI Access](https://www.cancer.gov/ccg/access-data)
* **Feature Extraction:** Each tabular omics modality is encoded via an independent fully-connected linear layer.
* **Data Splits:** Extracted sequentially: **169** training (60%), **76** validation (25%), and **61** test (20%).

### ROSMAP
* **Data Source and Content:** Multi-omics data from post-mortem brain tissue for Alzheimer's disease research.
* **Link:** [GitHub Repository](https://github.com/txWang/MOGONET)
* **Feature Extraction:** Each tabular omics modality is encoded via an independent Identity mapping encoder.
* **Data Splits:** Extracted sequentially: **194** training (60%), **87** validation (25%), and **70** test (20%).

### SIIM-ISIC
* **Data Source and Content:** Dermoscopic images of skin lesions with patient-level metadata for melanoma classification (2020 Kaggle challenge).
* **Link:** [Kaggle Competition](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data)
* **Feature Extraction:** `ResNet` encoder for images; Identity mapping encoder for tabular metadata.
* **Data Splits:** We adopt an 8:1:1 split (Seed: 42): **26,502** training, **3,312** validation, and **3,312** test.

### Derm7pt
* **Data Source and Content:** Multiclass skin lesion classification based on the 7-point checklist. Contains dermoscopic images and semi-quantitative clinical features.
* **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/menakamohanakumar/derm7pt)
* **Feature Extraction:** Images encoded with `CNNEncoder`; tabular clinical features encoded with Identity mapping.
* **Data Splits:** Official split: **413** training, **203** validation, and **395** test.

### GAMMA
* **Data Source and Content:** Glaucoma grading dataset containing color fundus images and stereo-pairs of disc photos.
* **Link:** [Zenodo Record](https://zenodo.org/records/15119049)
* **Feature Extraction:** Both fundus images and stereo-pair images are encoded using a `CNNEncoder`.
* **Data Splits:** Extracted sequentially: **20** training (20%), **10** validation (10%), and **70** test (70%).

### MIMIC-III
* **Data Source and Content:** ICU database with structured data (lab results, vitals) and clinical notes for mortality prediction.
* **Link:** [PhysioNet](https://physionet.org/content/mimiciii/1.4/)
* **Feature Extraction:** Modality encoded by `TimeSeriesTransformerEncoder` and Identity mapping.
* **Data Splits:** Extracted sequentially: **24,462** training (75%), **1,631** validation (5%), and **6,523** test (20%).

### MIMIC-CXR
* **Data Source and Content:** Chest X-ray images and corresponding radiology reports.
* **Link:** [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
* **Feature Extraction:** `ResNet` encoder for X-rays; `BERT` for radiology reports.
* **Data Splits:** Official split (training capped at 5k, Seed: 42): **5,000** training, **2,942** validation, and **5,117** test.

### eICU
* **Data Source and Content:** Multi-center ICU database with high-granularity vital sign data and clinical notes.
* **Link:** [eICU Website](https://eicu-crd.mit.edu/https://physionet.org/content/eicu-crd/2.0/)
* **Feature Extraction:** All modalities are encoded using the identity mapping.
* **Data Splits:** Hierarchical partition (Seed: 42): **5,727** training (~75%), **382** validation (~5%), and **1,528** test (~20%).

### MIRFLICKR
* **Data Source and Content:** Images from Flickr with associated user-assigned tags, used for retrieval and classification.
* **Link:** [Official Site](https://press.liacs.nl/mirflickr/)
* **Feature Extraction:** `ResNet` for images; `BERT` for textual tags.
* **Data Splits:** We adopt a 7:1:2 split (Seed: 42): **14,010** training, **2,001** validation, and **4,004** test.

### CUB Image-Caption
* **Data Source and Content:** Detailed bird images with rich, descriptive text captions.
* **Link:** [GitHub Repository](https://github.com/iffsid/mmvae)
* **Feature Extraction:** `ResNet` for images; `BERT` for text captions.
* **Data Splits:** We adopt a 7:1.5:1.5 split (Seed: 2025): **82,510** training, **17,680** validation, and **17,690** test.

### SUN-RGBD
* **Data Source and Content:** Large-scale dataset for indoor scene understanding with RGB-D (color and depth) images.
* **Link:** [Project Page](https://rgbd.cs.princeton.edu/)
* **Feature Extraction:** `ResNet` for RGB images; similar `ResNet` for depth maps.
* **Data Splits:** Official split: **4,845** training and **4,659** test.
    * *Note:* Test set is reused as validation set.

### NYUDv2
* **Data Source and Content:** RGB and Depth images of indoor scenes captured from a Microsoft Kinect.
* **Link:** [Official Site](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)
* **Feature Extraction:** `ResNet` encoder for RGB; `ResNet` for depth.
* **Data Splits:** Official split: **795** training, **414** validation, and **654** test.

### UPMC-Food101
* **Data Source and Content:** Food images paired with ingredient lists.
* **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/gianmarco96/upmcfood101)
* **Feature Extraction:** We follow [MMBT](https://github.com/facebookresearch/mmbt) for splits. `ResNet` for food images; `BERT` for ingredient lists.
* **Data Splits:** Official split: **62,971** training, **5,000** validation, and **22,715** test.

### MNIST-SVHN
* **Data Source and Content:** Synthetic dataset combining MNIST (handwritten digits) and SVHN (street view house numbers).
* **Link:** [GitHub Repository](https://github.com/iffsid/mmvae)
* **Feature Extraction:** Flattening operation for MNIST; Simple CNN encoder for SVHN.
* **Data Splits:** Official split: **560,680** training and **100,000** test.
    * *Note:* Test set is reused as validation set.

### N-MNIST + N-TIDIGITS
* **Data Source and Content:** Neuromorphic versions of MNIST (vision) and TIDIGITS (audio-spoken digits) recorded as event streams.
* **Feature Extraction:** Following [this repo](https://github.com/MrLinNing/MemristorLSM). NMNIST frames encoded by `CNN`; NTIDIGITS MFCCs encoded by `LSTM`.
* **Data Splits:** Extracted sequentially: **2,835** training (70%), **603** validation (15%), and **612** test (15%).

### E-MNIST + EEG
* **Data Source and Content:** E-MNIST (handwritten letters/digits) paired with simultaneously recorded EEG brain signals.
* **Feature Extraction:** Following [this repo](https://github.com/MrLinNing/MemristorLSM). E-MNIST encoded with `CNN`; EEG signals encoded with `LSTM`.
* **Data Splits:** Extracted sequentially: **468** training (70%), **104** validation (15%), and **130** test (15%).
