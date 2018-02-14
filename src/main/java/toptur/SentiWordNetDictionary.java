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

    /**
     * Returns 1 if it's subjective, -1 if it's objective, and 0 if the word has not been seen before.
     *
     * @param word - word to check
     * @return int
     */
    public int isWordSubjective(String word) {
        if (dictionary.containsKey(word)) {
            if (dictionary.get(word).getObjectivity() < 0.5)
                return 1;
            else
                return -1;
        }

        return 0;
    }

    public double getObjectivityOf(String word) {
        if (dictionary.containsKey(word))
            return dictionary.get(word).getObjectivity();

        return 1.0; // Assume full objectivity
    }
}
