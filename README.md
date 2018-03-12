# Topaz Turtle

My project for information extraction. This README will explain more about it. Eventually.

## What do we have to do next?

- [x] Extract the opinion in the sentence
- [ ] Allow evaluation of a specific metric for test command
- [ ] Extract the agent in the sentence
- [ ] Extract the target in the sentence
- [ ] Figure out the polarity of the opinion
- [ ] Add another classifier--perhaps a set of decision stumps?
- [ ] Add a bunch of features

### Things Andre needs to do

- [x] Change objectivity value representation for LibLinear
- [x] Polarity of sentence
- [x] Debug it on the CADE machines
- [ ] Add a bunch of features
- [ ] Extract everything and print it out

## Progress

* I can detect with 53% accuracy whether or not a given sentence has an opinion simply by using "bag of words" features.
I'd really like to see better accuracy than that.
* Up to 55% with the addition of objectivity of sentence!
* So depending on what seed you use, you get a variety of accuracies. A seed of 2 provides much higher accuracy than 1992
* I wrote the evaluation method, which was super simple. I'm getting an FScore on sentences of 0.37855, which really isn't
bad--I was expecting a whole lot worse. On different seeds I do get a whole lot worse though.
* Turns out each sentence can easily have more than 1 opinion--at times, there are as many as 6. That's problematic. Here
are the counts:
    1. 124 sentences
    2. 97
    3. 42
    4. 14
    5. 4
    6. 3
We should modify the sentence gatherer accordingly.
* With the recent modifications I've done on my branch, all of this should be mitigated. Each opinion expression has
its own frame when placed in a JSON file, meaning a single sentence that has six opinions in it will have six different
frames to represent it. The opinion extractor will take a single sentence and return a collection of opinion expressions
it found inside.
* So I'm getting 46% accuracy on the sentiment step, both on CADE and on my local machine. So that needs to be patched up.

## Resources

### Stuff I'm using

* [DataMuse](http://www.datamuse.com/api/)
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/simple.html)
* [SentiWordNet](http://sentiwordnet.isti.cnr.it/)

### Stuff I'm NOT using (yet)

* [WordNet](https://wordnet.princeton.edu/)
* [NewsAPI](https://newsapi.org/)