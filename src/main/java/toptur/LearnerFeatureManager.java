package toptur;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public class LearnerFeatureManager {

    private int idCounter;
    private Map<LearnerFeature, Map<Object, Integer>> ids;
    private Map<LearnerFeature, Set<Object>> potentialValues;

    public LearnerFeatureManager() {
        idCounter = 0;
        ids = new HashMap<LearnerFeature, Map<Object, Integer>>();
        potentialValues = new HashMap<LearnerFeature, Set<Object>>();

        for (LearnerFeature f : LearnerFeature.values()) {
            ids.put(f, new HashMap<Object, Integer>());
            potentialValues.put(f, new TreeSet<Object>());
        }
    }

    /**
     * Returns the feature ID for object given some feature.
     *
     * @param feature -
     * @param value -
     * @return int
     */
    public int getIdFor(LearnerFeature feature, Object value) {
        Map<Object, Integer> map = ids.get(feature);

        if (!map.containsKey(value)) {
            map.put(value, idCounter++);
            potentialValues.get(feature).add(value);
        }

        return map.get(value);
    }

    /**
     * Returns the set of all potential values for some feature.
     *
     * @param f -
     * @return Set of objects
     */
    public Set<Object> potentialValuesFor(LearnerFeature f) {
        return potentialValues.get(f);
    }
}
