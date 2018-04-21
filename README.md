# Topaz Turtle

My project for information extraction. This README will explain more about it. Eventually.

## Multithreading

Here's all the run times for each of the different portions along with the number of threads used

* Pre-Processing
    * Getting words from DataMuse
        1. 219.811s
        2. 146.491s
        3. 133.012s
        4. 133.909s
        5. 124.777s
        6. 118.318s
        7. 114.461s
        8. 110.879s 
* Training
    * Decision Tree using all features of depth 1. There were 9501 different features that needed to be considered.
        1. 238.456s
        2. 120.49s
        3. 81.584s
        4. 63.203s
        5. 63.213s
        6. 64.372s
        7. 64.555s
        8. 64.801s
    * 50 Bagged Trees of depth 1, using only 100 features and 100 examples, and using x threads for each tree
        1. 35.692s
        2. 9.846s
        3. 9.616s
        4. 9.414s
        5. 9.894s
        6. 11.704s
        7. 15.559s
        8. 18.142s
    * 50 Bagged Trees of depth 1, 100 features and examples, and 4 threads for the decision tree
        1. 10.934s
        2. 9.301s
        3. 9.144s
        4. 8.929s
        5. 8.989s
        6. 9.157s
        7. 9.766s
        8. 9.394s
    * Cross Product stuff (x, y) --- where x = bagged tree threads and y = decision tree threads
        * BaggedTrees50D1T1,1     29.357s
        * BaggedTrees50D1T1,2     14.413s
        * BaggedTrees50D1T1,3     9.95s
        * BaggedTrees50D1T1,4     7.674s
        * BaggedTrees50D1T1,5     7.776s
        * BaggedTrees50D1T1,6     7.803s
        * BaggedTrees50D1T1,7     8.536s
        * BaggedTrees50D1T1,8     7.791s
        * BaggedTrees50D1T2,1     14.402s
        * BaggedTrees50D1T2,2     7.454s
        * BaggedTrees50D1T2,3     7.486s
        * BaggedTrees50D1T2,4     7.369s
        * BaggedTrees50D1T2,5     7.368s
        * BaggedTrees50D1T2,6     7.334s
        * BaggedTrees50D1T2,7     7.548s
        * BaggedTrees50D1T2,8     7.457s
        * BaggedTrees50D1T3,1     9.816s
        * BaggedTrees50D1T3,2     7.423s
        * BaggedTrees50D1T3,3     7.226s
        * BaggedTrees50D1T3,4     7.336s
        * BaggedTrees50D1T3,5     7.957s
        * BaggedTrees50D1T3,6     7.424s
        * BaggedTrees50D1T3,7     7.343s
        * BaggedTrees50D1T3,8     7.813s
        * BaggedTrees50D1T4,1     9.552s
        * BaggedTrees50D1T4,2     7.372s
        * BaggedTrees50D1T4,3     7.851s
        * BaggedTrees50D1T4,4     7.426s
        * BaggedTrees50D1T4,5     7.479s
        * BaggedTrees50D1T4,6     7.413s
        * BaggedTrees50D1T4,7     7.509s
        * BaggedTrees50D1T4,8     7.422s
        * BaggedTrees50D1T5,1     7.621s
        * BaggedTrees50D1T5,2     7.389s
        * BaggedTrees50D1T5,3     7.536s
        * BaggedTrees50D1T5,4     7.525s
        * BaggedTrees50D1T5,5     7.458s
        * BaggedTrees50D1T5,6     7.422s
        * BaggedTrees50D1T5,7     7.356s
        * BaggedTrees50D1T5,8     7.648s
        * BaggedTrees50D1T6,1     7.486s
        * BaggedTrees50D1T6,2     7.522s
        * BaggedTrees50D1T6,3     7.632s
        * BaggedTrees50D1T6,4     7.469s
        * BaggedTrees50D1T6,5     7.529s
        * BaggedTrees50D1T6,6     7.547s
        * BaggedTrees50D1T6,7     7.32s
        * BaggedTrees50D1T6,8     7.536s
        * BaggedTrees50D1T7,1     7.554s
        * BaggedTrees50D1T7,2     7.509s
        * BaggedTrees50D1T7,3     7.576s
        * BaggedTrees50D1T7,4     7.496s
        * BaggedTrees50D1T7,5     7.661s
        * BaggedTrees50D1T7,6     7.524s
        * BaggedTrees50D1T7,7     7.438s
        * BaggedTrees50D1T7,8     7.507s
        * BaggedTrees50D1T8,1     7.581s
        * BaggedTrees50D1T8,2     7.574s
        * BaggedTrees50D1T8,3     7.726s
        * BaggedTrees50D1T8,4     7.358s
        * BaggedTrees50D1T8,5     7.689s
        * BaggedTrees50D1T8,6     7.607s
        * BaggedTrees50D1T8,7     7.548s
        * BaggedTrees50D1T8,8     7.605s
    * The best one was 3,3, which is reasonable to me
    * The whole enchilada, before incorporating the main thread:
        1) 1 Thread
            * BaggedTreesOpinions     418.679s
            * BaggedTreesPolarity     15.904s
            * BaggedTreesSentences    12.963s
            * PreProcessing   2.133s
            * TrainAgent      72.324s
            * TrainTarget     58.534s
            * TrainingAll     578.752s
        2) 2 Threads
            * BaggedTreesOpinions     391.054s
            * BaggedTreesPolarity     19.231s
            * PreProcessing   2.27s
            * TrainAgent      197.019s
            * TrainTarget     134.164s
            * TrainingAll     391.375s    
        3) 3 Threads
            * BaggedTreesOpinions     521.696s
            * BaggedTreesPolarity     43.394s
            * DataMuse        113.013s
            * PreProcessing   115.444s
            * TrainAgent      -0.001s
            * TrainTarget     182.735s
            * TrainingAll     522.008s


## Resources

### Stuff I'm using

* [DataMuse](http://www.datamuse.com/api/)
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/simple.html)
* [SentiWordNet](http://sentiwordnet.isti.cnr.it/)

### Stuff I'm NOT using (yet)

* [WordNet](https://wordnet.princeton.edu/)
* [NewsAPI](https://newsapi.org/)