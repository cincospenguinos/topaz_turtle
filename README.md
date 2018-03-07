# Topaz Turtle

My project for information extraction. This README will explain more about it. Eventually.

## What do we have to do next?

- [ ] Extract the opinion in the sentence
- [ ] Extract the agent in the sentence
- [ ] Extract the target in the sentence
- [ ] Figure out the polarity of the opinion
- [ ] Add another classifier--perhaps a set of decision stumps?
- [ ] Add a bunch of features

## Progress

* I can detect with 53% accuracy whether or not a given sentence has an opinion simply by using "bag of words" features.
I'd really like to see better accuracy than that.
* Up to 55% with the addition of objectivity of sentence!
* So depending on what seed you use, you get a variety of accuracies. A seed of 2 provides much higher accuracy than 1992
* I wrote the evaluation method, which was super simple. I'm getting an FScore on sentences of 0.37855, which really isn't
bad--I was expecting a whole lot worse. On different seeds I do get a whole lot worse though.

### Features that do not help detect opinions:

* Contains bigram
* Contains organization
* Contains person
* Contains ANY named entity (Drops to 50%)
* Starts with trigram

## Resources

### Stuff I'm using

* [DataMuse](http://www.datamuse.com/api/)
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/simple.html)
* [SentiWordNet](http://sentiwordnet.isti.cnr.it/)

### Stuff I'm NOT using (yet)

* [WordNet](https://wordnet.princeton.edu/)
* [NewsAPI](https://newsapi.org/)