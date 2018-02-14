# Topaz Turtle

My project for information extraction. This README will explain more about it. Eventually.

## Progress

* I can detect with 53% accuracy whether or not a given sentence has an opinion simply by using "bag of words" features.
I'd really like to see better accuracy than that.
* Up to 55% with the addition of objectivity of sentence!

### Features that do not help detect opinions:

* Contains bigram
* Contains organization
* Contains person
* Contains ANY named entity (Drops to 50%)
* Starts with trigram

## Resources

* [DataMuse](http://www.datamuse.com/api/)
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/simple.html)
* [WordNet](https://wordnet.princeton.edu/)