package toptur;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public class LearnerFeatureManager {

    private volatile int idCounter;
    private Map<LearnerFeature, Map<Object, Integer>> ids;
    private Map<LearnerFeature, Set<Object>> potentialValues;
    private Map<Integer, LearnerFeature> idsToFeatureTypes;

    private static LearnerFeatureManager instance;

    public static LearnerFeatureManager getInstance() {
        if (instance == null)
            instance = new LearnerFeatureManager();

        return instance;
    }

    private LearnerFeatureManager() {
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
            int id = idCounter;
            map.put(value, id); // TODO: Manage the concurrency issue here
            potentialValues.get(feature).add(value);
            idsToFeatureTypes.put(id, feature);
            idCounter += 1;
        }

        return map.get(value);
    }

    public Set<Object> potentialValuesFor(int featureId) {
        if (idsToFeatureTypes.containsKey(featureId))
           return potentialValuesFor(idsToFeatureTypes.get(featureId));

        throw new RuntimeException("No valid feature for ID " + featureId); // This will probably be thrown sometime soon
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
