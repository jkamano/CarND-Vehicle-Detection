
[//]: # (Image References)
[image0]: ./examples/pipeline.png
[image1]: ./examples/HLS_hog_vehice.png
[image2]: ./examples/HLS_hog_nonvehice.png
[image11]: ./examples/YCrCb_hog_vehice.png
[image21]: ./examples/YCrCb_hog_nonvehice.png
[image3]: ./examples/roi.png

[image05b]: ./examples/boxes_05.png
[image05h]: ./examples/boxes_05_heat.png
[image1b]: ./examples/boxes_1.png
[image1h]: ./examples/boxes_1_heat.png
[image15b]: ./examples/boxes_15.png
[image15h]: ./examples/boxes_15_heat.png
[image2b]: ./examples/boxes_2.png
[image2h]: ./examples/boxes_2_heat.png
[image115b]: ./examples/boxes_1and15.png
[image115h]: ./examples/boxes_1and15_heat.png

[image4]: ./examples/test1_box.png
[image5]: ./examples/test2_box.png
[image6]: ./examples/test3_box.png

[video1]: ./output.mp4

#Vehicle Detection Project

![alt text][image0]
**Fig.1** Pipeline used for training and detection of cars.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code to extract HOG features for training is in function `get_hog_features()` in `train.py line 130`

I started by reading in all the `vehicle` and `non-vehicle` images and assigning tags to it, `1` for vehicle and `0` for non-vehicle images. 
The data was then randomly shuffled and divided into 20% for test and 80% for training.
A total of 14208 images were used for training and  3552 for testing.

Feature extraction is done in `extract_features()` function on `train.py line 31` and 
Several combinations of features were tested.
Regarding the HOG features, at first the color space was HLS and using only the L channel as that one seem to contain much more edge information than the others.
The results, fitting an linear SVM were around 98.5% correct recognition on the test set.
Later following some cues on the class, used the 3 channels of YCrCb and it boosted a little the recognition to 99.16%.
The combinations of features tested are shown below, the accuracy improvements are marginal but an effort was made to keep number of features as low as possible while keeping accuricy above 99%.

The feature extraction for Detection is shown to be shared between Train and Detection in the Block Diagram but in fact, for Detection, the function extracting feature is `find_cars()` on line 61 of `detect.py`.
The reason is the optimization of hog feature extraction, 

####2. Explain how you settled on your final choice of HOG parameters.

First plotted images for `vehicle` and `nonvehicle` images with resulting HOG features for each channel and for two color spaces `YCrCb` and `HLS` since it is usefull to separate chroma and luminance.
This is done in `train.py line 173`
```python
vis = False #Visualize hog features for a random sample
train = True #If False subsamples the data set to speedup execution
```

||Vehicle|Non-Vehicle|
|-|-|-|
|HLS|![alt text][image1]|![alt text][image2]|
|YCrCb|![alt text][image11]|![alt text][image21]|

Knowing what feature extraction is doing, I tried several combinations of parameters.
HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` were kept constant.
The following table shows the tested combinations, the last line is the selected combination of parameters because minimizes number of features while keeping accuracy above 99%.

| Colorspace| ColorHist| Spacial| HOG_ch |HOG_orient|HOG_ppcel|HOG_cpblock| Tot # Features |Accuracy|
| ------------ |:--------------:| --------:|-----:|--:|---:|--:|---------:|----------:|
| YCrCb      | 32binsx3ch | 32x32 |  All|  9|  8|  3|  11916|  99.16 %|
| YCrCb      | 32binsx3ch | 32x32 |  Y  |  9|  8|  3|    6084|  98.68 %|
| HLS         | 32binsx3ch | 32x32 |  All|  9|  8|  3|  11916|  99.07 %|
| HLS         | 32binsx3ch | 32x32 |  L   |  9|  8|  3|    6084|  98.45 %|
| YCrCb      | 32binsx3ch | 32x32 |  Y  |  9|  8|  2|    4932|  98.48 %|
| YCrCb      | 32binsx3ch | 32x32 |  All|  9|  8|  2|    8460|  99.27 %|
| YCrCb      | 32binsx3ch | 16x16 |  All|  9|  8|  2|    6156|  99.32 %|

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using sklearn functions, as stated above with a training set of 14208 images.
For each 64x64 image the features extracted are:

* Spatial features: 3 channels * 16px * 16px = 768
* HOG features: 3 channels * 7 blockpositions * 7 blockpositions * 2 cpblock * 2 cpblock * 9 orient = 5292
* Color Histogram features: 3 channels * 32bins = 96 

This is done at the very end of `train.py` (line 173 downwards)

At the end of training, the svm classifier object is saved for later use in `detect.py`. The scaler used to normalize the different features in the feature vector is also stored so that on runtime the algorithm know how it scaled the features during training. It is very important that the scaling factors are the same and that the features keep the same order in the vector.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the purpsed function to slide across an region of interest (ROI). The ROI chosen contains full horizontal image but vertically just from pixel 380 to 656, limit to 656 cuts our own car hood, 380 is cutting the sky and tree tops.
![alt text][image3]
To optimize scales to search and threshold on the heatmap, several scales and window overlay (cells to step) were tested. Tried to optimize separation between cars, trying to get the windows as tight as possible to the vehicle but with minimum number of scales to reduce computational effort.

|Parameters| Boxes | Heatmap |
| ------------- |: ------- :| ------------:|
|Scale=0.5  |![alt text][image05b]|![alt text][image05h]|
|Scale=1  |![alt text][image1b]|![alt text][image1h]|
|Scale=1.5  |![alt text][image15b]|![alt text][image15h]|
|Scale=2 step=1  |![alt text][image2b]|![alt text][image2h]|
|Scale=1+1.5  |![alt text][image115b]|![alt text][image115h]|


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:
(exactly) =)

|![alt text][image4]|![alt text][image5]|
|--|--|
|![alt text][image6]||

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The pipeline to process video frames is implemented in the class `pipeline()` method `process()` line 301 of `detect.py`.
The frame is extracted from the video on line 308 of `detect.py`.
An instance of pipeline class `pipe` is initialized with the trained classifier and scaler, saved in `train.py`.
This pipeline instance is than called for every frame, the instance stores not only classifier and scaler (constants) but also variables that are persistent between several frame processing.

The process explained above, with sliding window and heat maps is used to obtain a heat map for each frame.
The function `scipy.ndimage.measurements.label()` is then used to identify individual blobs in the heatmap and create bounding boxes around them. Each bounding box (BBox) has a center and an area, trivially computed.

To assign these BBoxes to vehicles, there is the function `assign()`, for each BBox detected it checks if it falls inside a vehicle or not. A vehicle is an object of the class `vehicle()` and it is stored in a list which is stored for the entire duration of the video.
There are 2 options to assign a BBox:

* The BBox center falls inside the area of a vehicle -> vehicle box is updated to the BBox, vehicle `age` increments and `kill_deb` counter resets.
* The BBox does not fall inside any vehicle -> Start a new vehicle object, with `age=0` and `kill_deb=0`

A vehicle box will only be visible if its `age` is above a certain threshold, to avoid false positives.

Each frame increments `kill_deb` of each vehicle in the list and if it is not reset by updating it with a newly detected BBox, this counter incremets, if it reaches a limit, vehicle is removed from the list (Vehicle disapears). This avoids false negatives. 

This is just a simple time debounce filter, more ellaborate techniques could and should probably be used like kalman filter tracking each vehicle where bounding boxes give noise measurements of the vehicle location and number/area of these boxes gives some degree of certainty.
Making also use of the apriori knowledge that a vehicle under reasonable temperatures, can not evaporate.

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Pipeline has some shortcomings, it loses track of vehicles in some frames. It also does not track hidden vehicle by occlusion. This is expected but could be improved using predictive model for each vehicle object.

* **False negatives**, some frames lose track of one of the vehicles. Like mentioned above, a more advance filter making use of a more precise model about expected vehicle movement on the frame could help avoid loss of tracking.

* **Identification**, when tracking a vehicle with a more advance method one of the issues will be assignment of it's ID to know to which filter instance to feed each measurement. This could be accomplished by using some of the features already computed for each window to keep a set of caracteristics of each tracked object.
Then if object is lost and found again it in not considered a new one but its characteristics may reveal it's original ID and continue its tracking. This assignment could be done finding some distance function between object characteristics and findind the optimal assignment e.g. with hungarian method.

* **Model object and predict**, other feature using a more advances filter would be a prediction of where object will be on the next frame. This could reduce the search region by a great factor. New objects would be searched for randomly or only on some frames. 

