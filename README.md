# Co-Speech Gesture Generator for 18 Joints

This is the project for Li Xiangqi Year Four Dissertation Project in Heriot-Watt University.

This system generates the co-speech gestures given transcipts for 18 joints in upperbody.

This is an implementation of *Robots learn social skills: End-to-end learning of co-speech gesture generation for humanoid robots* ([Paper](https://arxiv.org/abs/1810.12541), [Project Page](https://sites.google.com/view/youngwoo-yoon/projects/co-speech-gesture-generation)) using [Talking With Hands 16.2M](https://github.com/facebookresearch/TalkingWithHands32M) for [GENEA Challenge 2022](https://genea-workshop.github.io/2022/challenge/).

## System Operation Overview

System input: .tsv file for transcripts of speech (every word should be denoted with start_time and end_time.

System output: .bvh file which has rotation matrices for 18 upper-body joints.

### Environment

The code was developed using python 3.7 on Windows.

For training, Google Colab is used with `co-speech-18joints.ipynb`.

Pytorch version: 1.13.1

CUDA version: 11.7

Other details for envrionment please check `requirements.txt`.

## Instructions

### Prepare

1. Install dependencies

   ```
   pip install -r requirements.txt
   ```
2. Download the FastText vectors from [here](https://fasttext.cc/docs/en/english-vectors.html) and put `crawl-300d-2M-subword.bin` to the resource folder (`resource/crawl-300d-2M-subword.bin`).
3. Unzip the dataset directory `mnt` and place it under the main repo.

### Train

1. Make LMDB. Here the path should be `..\mnt\trn\`.

   ```
   cd scripts
   python twh_dataset_to_lmdb.py [PATH_TO_DATASET]
   ```
2. If using Windows for training, update paths and parameters in `PROJECT_ROOT/config/seq2seq.yml` and run `train.py`

```
python train.py --config=../config/seq2seq.yml
```

3. If using Google COLAB, update paths and parameters in `config/seq2seq.yml`. Open Google COLAB and run `co-speech-18joints.ipynb` block by block.

### Inference

1. Pretrained model from the author and the checkpoint file trained in this project are both provided.

   * Pretrained model: `output/train_seq2seq/baseline_icra19_checkpoint_100.bin`
   * Trained model from this project: `output/train_seq2seq/baseline_lixiangqi_final_checkpoint_100.bin`
2. Put [vocab_cache.pkl](https://www.dropbox.com/s/fif332qp00e5qly/vocab_cache.pkl?dl=0) file into lmdb train path.
3. Output a BVH motion file from speech text (TSV file).

   ```
   python inference.py [PATH_TO_MODEL_CHECKPOINT] [PATH_TO_TSV_FILE]
   ```

### Evaluation

The evaluation code is cloned and adapted from [numerical evaluation of GENEA Gesture Generation Challenge 2022](https://github.com/genea-workshop/genea_numerical_evaluations/tree/2022?tab=readme-ov-file). Please see readme.md in `evaluation` directory for details.


Visualization

The visualization code is cloned from [here](https://github.com/TeoNikolov/genea_visualizer/tree/archive_2022). Please use the `blender_render.py` for visualization in Blender.
