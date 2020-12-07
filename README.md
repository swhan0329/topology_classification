# topology_classification
This is a topology classification training and inference code using pretrained the ResNet.

When you input the image of topology, this network predicts how many nodes there are in this image.

The number of class is six. This means that an image with 3, 4, 5, 6, 7, and 8 nodes is used.

## Our data set configuration
Data Set is made by Yoon Kyung Jang (jokjjs0216@khu.ac.kr)

Image(.png) + label(.txt) -> preprocessing -> image+label(.json)

Data set link: https://drive.google.com/file/d/1gqCReEtbO1N2glrRoKS4yF1wi-bjBQY0/view?usp=sharing

The number of data set: 1854

### Example of data set
![ex_screenshot](./input/14.png =100x100)

![ex_screenshott](./input/104.png =100x100)

## Network
### Pretrained Weight File
Link: 

## How to run this code
0. Preprocess

```bash
python create_data.py \
-i [input image directory] #(<image input dir>/*.png) \
-l [label information in text file] #(<label input dir>/*.txt) \
-o [output directory]
```

1. Train and validation

```bash
python main.py
```

2. Test(Inference)

```bash
python inference.py -i [input image] -w [weight file]
```

## Result

### Best Accuracy
__Train accuracy:__ 
__Validation accuracy:__ 

### Train vs. Validation accuracy graph


## Q & A
If you have a question, please send e-mail to me.

swhan0329@gmail.com
