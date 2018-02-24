package toptur;

import java.util.HashMap;

/**
 * A single dictionary that contains all of the words from SentiWordNet.
 *
 * http://sentiwordnet.isti.cnr.it/
 */
public class SentiWordNetDictionary {

    private HashMap<String, SentiWordValues> dictionary;


    public SentiWordNetDictionary() {
        dictionary = new HashMap<String, SentiWordValues>();
    }

    public void addWord(String word, double positive, double negative) {
        if (dictionary.containsKey(word)) {
            dictionary.get(word).includeNewValues(positive, negative);
        } else {
            dictionary.put(word, new SentiWordValues(positive, negative));
        }
    }

    public boolean hasWord(String word) {
        return dictionary.containsKey(word);
    }

    public double getObjectivityOf(String word) {
        if (dictionary.containsKey(word))
            return dictionary.get(word).getObjectivity();

        return 1.0; // Assume full objectivity
    }
}
