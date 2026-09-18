Директория для обучающих данных <br/>
Предполагаемая иерархия: <br/>

![alt text](../info_images/sample_traindata_directory.png)

Обучающие изображения с параметрами должны находиться в training_data/dd.mm.yyyy/images/for_predictor/unprepared <br/>
Обучающие видео должны находиться в training_data/dd.mm.yyyy/videos <br/>

Изображения с параметрами и изображения с видео разделены. <br/>
Это сделано из-за того, что они могли быть получены из разных видео, <br/> следовательно, свечения на этих изображениях могут находиться в разных местах на изображениях, <br/> а это может усложнить процесс обрезки видео. 

Directory for all the data <br/>
Supposed hierarchy view: <br/>

![alt text](../info_images/sample_traindata_directory.png)

Images with parameters goes to training_data/images/for_detector/unprepared <br/>
Videos goes just to training_data/videos

images with parameters and without are separated <br/> 
because they can be extracted from different videos <br/> 
and its general cropping can be difficult