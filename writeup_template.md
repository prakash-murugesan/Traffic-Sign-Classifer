# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.



Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data: 


![Classes](https://github.com/prakash-murugesan/Traffic-Sign-Classifer/blob/master/training%20set%20distribution.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale however I read the scientific paper Yann LeCun used and noticed that they transformed the images to the YUV space. However, with further exploration I realized that using 3 1x1 convolutions allows the net to learn it's own color space transform that works the best. So I left the data in the color space and used the 3 1x1 convolutions and it gave a huge improvement in the validation accuracy 


I normalized the image data because it allowed the data to converge faster. Here's an image after normalization..be warned its spooky!

![Normalized boogeyman](https://github.com/prakash-murugesan/Traffic-Sign-Classifer/blob/master/traffic%20sign%20classifier%20of%20your%20nightmares.png)

I decided to generate additional data because the classes were skewed. It might reflect the natural occurence of the signs, such as a stop sign being more common than say "falling rocks". However, what this leads to is the classifier being biased to the more common signs. If it was faced with two choices and it was unsure, it would chose the sign it has seen more often. So to counteract this bias i created more data by jittering the images. 

To add more data to the the data set, I used opencv's function cv2.warpAffine to transform, shear, and rotate the images. I initially tried to have around 2000 examples for each class. However, augmenting the image took a lot of time and with time considerations, I put the threshold to 1000. The data set may still be slightly skewed but not to the extent of the original. 
![updated data](https://github.com/prakash-murugesan/Traffic-Sign-Classifer/blob/master/updated%20distribution.png)

However, what I noticed was that my validation accuracy had dropped to 91.7 % but my test accuracy was actually better than the validation accuracy at 93%! Perhaps, this was because I was jittering the images too much. With time considerations and the fact that the model performed well without augmented data I decided to discard that approach.  

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:
![CNN Model](https://github.com/prakash-murugesan/Traffic-Sign-Classifer/blob/master/CNN%20model.PNG)

I began with the LeNet model, however through various online research and through my own curiosity I kept tweaking it over a weekend to arrive at what I think is somewhat an efficient way. I tried different forms and finally settled down to the "long-thin" model as the way i describe it. I used 3x3 convolutions and doubled the number of filters for each layer starting from 16 to 128. Finally I connected it to two fully connected layers of 120 and 83, ultimately with 43 outputs representing the 43 unique classes we are in search of. 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I went with the starting base parameters mostly. They seemed to yield good enough results. However, I lowered my learning rate to 0.00075 as I found that the model started bouncing around a local minima quite early in the training. I had the time to go for a 100 epochs and wanted to see some more progression. On the other hand, counter intuitively I found that as I increased the batch size to 512 the validation accuracy dropped. So I went back to 128 and trained the Neural net. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model had a validation set accuracy of 97.8% and a test set accuracy of 96.7%. I believe I was able to reach a good test set accuracy levels that closely matched the validation accuracy by the heavy use of dropouts between the layers. This allowed the model to generalize better. 

My approach was an iterative one as all things should be. I hoped to get better with each try, although I must admit that some models in the middle had better performance. But my experience of continuing to tweak the model helped me develop a better intuition of how these neural nets worked. 

Although I began with the LeNet architecture, I started incorporating ideas that I'd read from around the internet for image recognition tasks. One interesting idea was to use 3 1x1 initial convolutional filters to learn the best color space for the input images instead of transforming the images to a YUV space or grayscale. I tried this with amazing results, an almost immediate jump in accuracy. However, I'm not sure but this might have come back to bite me on the external test images. 

I heavily used dropouts between layers, I hypothesize that being the reason why my test accuracy matches so closely with my validation accuracy.  

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are the Six German traffic signs that I found on the web:

![1.](https://github.com/prakash-murugesan/Traffic-Sign-Classifer/blob/master/test_images/1.png)![2.](https://github.com/prakash-murugesan/Traffic-Sign-Classifer/blob/master/test_images/2.png) ![3.](https://github.com/prakash-murugesan/Traffic-Sign-Classifer/blob/master/test_images/3.png) ![4.](https://github.com/prakash-murugesan/Traffic-Sign-Classifer/blob/master/test_images/4.png) ![5.](https://github.com/prakash-murugesan/Traffic-Sign-Classifer/blob/master/test_images/5.png) ![6.](https://github.com/prakash-murugesan/Traffic-Sign-Classifer/blob/master/test_images/6.png)

I went for images generally spread around the data set. Picking some from the skewed classes to see if it made a difference. I also chose signs that looked relatively similar to the others to see if it may confuse the image recognizer. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 83.3%. However, the test set accuracy was much better at almost 97%. I'm not exactly sure why it wasn't performing as well, however it may have to do with the different approach I took of using all 3 color channels without grayscale transformation through the 3 1x1 filters. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model is 100% sure on the 3 out of the 5 models and gets them right. It is 98% sure on the triangle shaped sign and just loses 2% to another relatively similar one. 

However, the biggest issue seems to be the sign that it actually ends up getting wrong. The model is very confident, 93% of the empty no vehicles sign being a no entry sign. The correct answer appears only in the 5th position with a rounded confidence of 0%. The model, I think, puts a bigger emphasis on the color band around these signs. The 5 softmax images all have bands around the edges and the second option isn't even circular! So perhaps, a better explanation can be arrived from checking out the intermediate convolution visualizations for this image. For now, I believe the reason why it does so poorly with this example is the relative similarity to the "no entry" sign compounded with the fact that the occurence of the "no entry" sign is higher than the "no vehicles" sign. The skewed data set may be at play here. While, it is a problem that could have been fixed early on, it seems my gamble for the short term gain of higher test/validation accuracy didn't translate so well to real world data. 

![Softmax Probabilities](https://github.com/prakash-murugesan/Traffic-Sign-Classifer/blob/master/softmax.PNG)

