package toptur;

import java.util.HashMap;
import java.util.TreeMap;

public class LibLinearFeatureManager {

    public enum LibLinearFeature { CONTAINS_UNIGRAM, OBJECTIVITY_OF_SENTENCE}

    private static volatile LibLinearFeatureManager instance;

    private int counter;
    private HashMap<LibLinearFeature, TreeMap<Object, Integer>> ids;

    private LibLinearFeatureManager() {
        counter = 1;
        ids = new HashMap<LibLinearFeature, TreeMap<Object, Integer>>();

        for (LibLinearFeature f : LibLinearFeature.values())
            ids.put(f, new TreeMap<Object, Integer>());
    }

    public static LibLinearFeatureManager getInstance() {
        if (instance == null)
            instance = new LibLinearFeatureManager();

        return instance;
    }

    /**
     * Returns liblinear ID for some LibLinearFeature.
     *
     * @param feature - Enumerated feature type
     * @param value - The value of the feature
     * @return ID representing that value for the given feature
     */
    public int getIdFor(LibLinearFeature feature, Object value) {
        TreeMap<Object, Integer> map = ids.get(feature);
        if (map.containsKey(value))
            return map.get(value);

        int id = counter;
        map.put(value, counter++);

        return id;
    }
}
