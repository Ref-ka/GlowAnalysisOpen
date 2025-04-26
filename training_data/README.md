Directory for all the data <br/>
Supposed hierarchy view: <br/>

* training data
* * dd.mm.yyyy
* * * images
* * * * for_detector
* * * * * prepared
* * * * * preprocessed_1080
* * * * * unprepared
* * * * for_predictor
* * * * * prepared
* * * * * preprocessed_1080
* * * * * unprepared
* * * videos

Images with parameters goes to training_data/images/for_detector/unprepared <br/>
Videos goes just to training_data/videos

images with parameters and without are separated <br/> 
because they can be extracted from different videos <br/> 
and its general cropping can be difficult