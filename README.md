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

## Training the Model

1. **Prepare the dataset:**
    Ensure your dataset is in the correct format and placed in the appropriate directory. You can create a custom dataset by following the instructions below.

2. **Train the model:**
    ```bash
    python train.py --data_dir path/to/dataset --output_dir path/to/save/models
    ```

3. **Monitor training:**
    Optionally, you can use TensorBoard to monitor the training process:
    ```bash
    tensorboard --logdir path/to/logs
    ```

## Testing the Model

1. **Evaluate the trained model:**
    ```bash
    python test.py --model_path path/to/save/models --data_dir path/to/testset
    ```

2. **View results:**
    The test script will output the evaluation metrics and any relevant visualizations.

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
