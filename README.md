# HiFi-CS: Hierarchical FiLM ClipSeg For Enhanced Vision Language Grounding In Robotic Grasping

## Abstract
Robots interacting with humans through natural language can unlock numerous applications such as Referring Grasp Synthesis (RGS), to generate a grasp pose for an object specified by a text query. RGS comprises two steps: visual grounding and grasp pose estimation. This paper introduces HiFi-CS, featuring hierarchical application of Featurewise Linear Modulation (FiLM) to fuse image and text embeddings, enhancing visual grounding for complex text queries. Visual grounding associates an object in 2D/3D space with the natural language input and is studied in two scenarios: Closed Vocabulary (environments with a fixed set of known objects) and Open Vocabulary (previously unseen  objects). Recent studies leverage powerful Vision-Language Models (VLMs) for visually grounding natural language in real-world robotic execution. However, comparisons in complex, cluttered environments with multiple instances of the same object categories have been lacking.  HiFi-CS features a lightweight decoder combined with a frozen VLM and outperforms competitive baselines in closed vocabulary settings. Additionally, our model can be used with open-set object detectors like GroundedSAM to enhance open-vocabulary performance. We validate our approach through real-world RGS experiments using a 7-DOF robotic arm, achieving a 90.33\% visual grounding accuracy in 15 complex tabletop scenes.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/vineet2104/HiFiCS-VisualGrounding.git
    cd HiFiCS-VisualGrounding/
    ```

2. **Create and activate a conda environment:**
    ```bash
    conda env create -f hifics_env.yml
    conda activate hifics
    ```

3. **Install CLIP**
    ```bash
    pip install git+https://github.com/openai/CLIP.git
    ```

## Training HiFi-CS

We offer support to train the model on 2 datasets - [RoboRefIt](https://ieeexplore.ieee.org/document/10341379) and [OCID-VLG](https://openreview.net/pdf/25fe155e277cb95267cd9b875bb02f9c88dcb8c5.pdf). Please follow the steps below to download the datasets and save it in the required format for easy training and testing. 

1. **Download the RoboRefIt dataset:**
   
    Download the dataset from [link](https://drive.google.com/file/d/1pdGF1HaU_UiKfh5Z618hy3nRjVbq_VuW/view?usp=sharing) and move it to the ./datasets/ folder

2. **Download and prepare the OCID-VLG dataset:**

    **A. Download the dataset:**
    Download the dataset from [link](https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view?usp=share_link) and move it to the `./datasets/` folder.

    **B. Process the dataset:**
    Run the following command to process the data and prepare train-test splits:
    ```bash
    python datasets/prepare_ocidvlg.py
    ```

    At this point, the `./datasets/` folder should look like:

    ```plaintext
    ./datasets/
    ├── dataloader.py
    ├── load_ocidvlg.py
    ├── prepare_ocidvlg.py
    ├── OCID-VLG/
    └── RoboRESTest/
    └── final_dataset/
        ├── testA/
        ├── testB/
        ├── train/
            ├── depth/
            ├── depth_colormap/
            ├── image/
            ├── mask/
            roborefit_train.json
    └── ocidvlg_final_dataset/
        ├── test/
        ├── train/
            ├── depth/
            ├── image/
            ├── mask/
            ocid_vlg_train.json
    ```

4. **Train HiFi-CS on RoboRefIt:**
    ```bash
    python training.py config.yaml 0
    ```

5. **Train HiFi-CS on OCID-VLG:**
    ```bash
    python training.py config.yaml 1
    ```

## Testing the Model

Trained models are saved in ./logs folder. After training HiFi-CS on RoboRefIt and OCID-VLG separately, we offer support to evaluate these models on their respective test sets.

1. **Evaluate hifics-roborefit-default on RoboRefIt Test A and Test B splits:**
    ```bash
    python score.py config.yaml 0 0
    ```
    Note - change the test_split variable in test_configuration to evaluate on testA or testB

2. **Evaluate hifics-ocidvlg-default on OCID-VLG Test split:**
    ```bash
    python score.py config.yaml 1 1
    ```

## Creating a Custom Dataset

1. **Collect your data:**
    Gather your data and organize it into appropriate folders, typically `train`, `val`, and `test`.

2. **Format your data:**
    Ensure your data is in the required format (e.g., images, annotations, etc.). Refer to the dataset preparation guidelines in the documentation.

3. **Update the configuration:**
    Modify the dataset configuration file to point to your custom dataset directories:
    ```json
    {
      "train": "path/to/train",
      "val": "path/to/val",
      "test": "path/to/test"
    }
    ```

4. **Preprocess the data:**
    If necessary, run any preprocessing scripts provided to prepare your data for training:
    ```bash
    python preprocess.py --data_dir path/to/dataset
    ```

## Citation

If you find our work useful in your research, please cite our paper:
```bibtex
@article{yourpaper2024,
  title={Title of Your Paper},
  author={Your Name and Co-Authors},
  journal={Journal Name},
  year={2024},
  volume={xx},
  number={xx},
  pages={xx-xx},
  doi={xx/xx/xx}
}
