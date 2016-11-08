Matlab files for Support Vector Machine Regression based on HOG features.  
##Training  
Run `train.m` for training.
+ Convert image to YUV.   
+ For each pixel, extract HOG features around this pixel (cell-size: 8, block-size:2).  
+ For each pixel, construct feature vector as all HOG features plus y value.   
+ Train a SVR model for U colorspace, using true U value as target and feature vector as features.   
+ Train a SVR model for V colorspace, using true V value as target and feature vector as features.  
##Validation  
Run `validation.m` for validation.
+ Convert image to YUV.  
+ Extract HOG features around each pixel and construct feature vector for the pixel using HOG features together with Y value.  
+ Predict U,V value.
+ Calculate error based on true U,V value.  
+ Average all pixel errors to get validation error.  
##Visualization  
Run `predict.m`. Remember to specify the path of the target image.  

By Zeyu Zhao.
Nov. 08, 2016
