# Baby-Crying-Detection![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)



## Introduction 

This despository is used for judging why baby wa crying?  In the early statge, the cry of baby has some common reasones. Under this background, we collect and slect some database  that includes discomfort, sleep, hunger, wakefulness, and hug five types. Finally ,we develop a  method which has good accuarcy in baby crying reason judgement. 




## How to use ?


1. Download some cry data:  Most of all the data have been proved by   [chris](https://www.kaggle.com/chris0223/babycry).

2. Download this code despository. put database into  wavs_all_cut. Test data into data/test.

3.   developing python environment for train.

4. run WriteMfcc2CSV.py for   train data generate. 


5. run ModelTrain.py for  model train.

6. run predict.py for test prediction. 

7. get the result. 





note : due time reason ,some  complex code is not in here. We give the DataAugument.py and some reference paper for your review. 

## reference

1.  aniel  S  Park,  William  Chan,  Yu  Zhang,  Chung-Cheng  Chiu,  BarretZoph,  Ekin  D  Cubuk,  and  Quoc  V  Le.   Specaugment:  A  simple  dataaugmentation method for automatic speech recognition.arXiv preprintarXiv:1904.08779, 2019.

2. Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz.mixup:  Beyond  empirical  risk  minimization.arXiv  preprintarXiv:1710.09412, 2017

3. hi Hua Zhou.  Ensemble learning.Encyclopedia of Biometrics, pages270–273, 2009.