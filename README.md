# STEP - Towards Structured Scene-Text Spotting

This repository contains the code and data for the paper [STEP - Towards Structured Scene-Text Spotting](https://arxiv.org/abs/2309.02356)

![STEP](figures/STEP.png)

## Running the Code

### Code and Environment Setup

Use the following commands to clone and create the environment with conda:

```
git clone https://github.com/CVC-DAG/STEP.git
cd STEP
conda create -n STEP python=3.8 -y 
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 cudatoolkit-dev=11.3 -c pytorch -c conda-forge
python -m pip install scipy numba
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
python setup.py build develop
```

### Datasets

Our proposed approach uses [HierText-based](https://github.com/google-research-datasets/hiertext) training 
and validation splits. The training and validation images can be downloaded using
the [AWS CLI interface](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html):

````
mkdir datasets
mkdir datasets/hiertext
aws s3 --no-sign-request cp s3://open-images-dataset/ocr/train.tgz datasets/hiertext
aws s3 --no-sign-request cp s3://open-images-dataset/ocr/validation.tgz datasets/hiertext
tar -xzvf datasets/hiertext/train.tgz -C datasets/hiertext/
tar -xzvf datasets/hiertext/validation.tgz -C datasets/hiertext/
````

Our pipeline uses custom training and validation ground truths. The ground truth files can be downloaded 
with the following script:

````
wget http://datasets.cvc.uab.cat/STEP/structured_ht.zip -P datasets/hiertext
unzip datasets/hiertext/structured_ht.zip -d datasets/hiertext
````

Finally, our proposed test set can be downloaded with:

````
wget http://datasets.cvc.uab.cat/STEP/structured_test.zip -P datasets
unzip datasets/structured_test.zip -d datasets
````

The license plate images are sourced from the [UFPR-ALPR](https://github.com/raysonlaroca/ufpr-alpr-dataset)
dataset. The images of this dataset are licensed for non-commercial use, you need to request access 
to the authors (instructions are included in the linked repository).
The images we used are the first frames of each one of the sequences, _script to copy the frames 
automatically coming soon_.

### Test Dataset Format

The dataset format follows the [labelme](https://github.com/labelmeai/labelme/tree/main) annotation
format. The "label" field of every annotation is its type/class of code (UIC, BIC, tonnage, etc.). The
field "transcription" contains the transcription of the instance. The following table specifies the 
label of every type of code and its regular expression:

| Class  | Regular Expression | Label |
| ------------- | ------------- | ------------- |
| BIC  | \\[A-Za-z]{4}\\s\\d{6}\\s\\d  | bic |
| UIC  | \\d{2}\\s?\\d{2}\\s?\\d{4}\\s?\\d{3}\\-\\d  | uic |
| TARE  | \\d{2}[.]?\\d{3}\\s?(?i)kg  | tare |
| Phone Num.  | \\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4} | phone |
| Tonnage  | \\d{2}[.]?\\d?[t] | tonnage |
| License Plate  | \\[A-Z]{3}\\s\\d{4} | lp |

### Model Weights

The TESTR pretrained weights (which are used to initialise the model) and STEP's final weights can 
be downloaded with:
```
mkdir ckp
wget http://datasets.cvc.uab.cat/STEP/comingsoon.pth -P ckp
wget http://datasets.cvc.uab.cat/STEP/comingsoon.pth -P ckp
```

They should be placed under the ``ckp`` directory, although you can place them anywhere else, but you 
should change the arguments of the example script calls below.

## Running the Model

The model can be trained with the following script (needs the TESTR pretrained weights linked above):

```
python tools/train_net.py --config-file configs/STEP/hiertext/STEP_R_50_Polygon.yaml --num-gpus 2
```

To run the validation script:

```
python inference/eval.py --config-file configs/STEP/hiertext/STEP_R_50_Polygon.yaml --opts MODEL.WEIGHTS ckp/STEPv1_final.pth MODEL.TRANSFORMER.INFERENCE_TH_TEST 0.3
 ```

Finally, to run the test script on our proposed test dataset:

```
python inference/test.py --config-file configs/STEP/hiertext/STEP_R_50_Polygon.yaml --opts MODEL.WEIGHTS ckp/STEPv1_final.pth MODEL.TRANSFORMER.INFERENCE_TH_TEST 0.3
```

## Create Your Own Queries

Todo _soon_

## License

This repository is released under the Apache License 2.0. Check the [LICENSE](LICENSE) file, dawg.

## Acknowledgements

We thanks [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) training and inference framework 
and the authors of [TESTR](https://github.com/mlpc-ucsd/TESTR) for their code and work.
