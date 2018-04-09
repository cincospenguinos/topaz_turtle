package toptur;

import com.google.gson.Gson;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
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

    public static LearnerFeatureManager getInstance(String fileName) {
        if (instance == null) {
            File f = new File(fileName);

            if (f.exists()) {
                instance = pullFromFile(f);
            } else {
                instance = new LearnerFeatureManager();
            }
        }

        return instance;
    }

    private LearnerFeatureManager() {
        idCounter = 0;
        ids = new HashMap<LearnerFeature, Map<Object, Integer>>();
        potentialValues = new HashMap<LearnerFeature, Set<Object>>();
        idsToFeatureTypes = new HashMap<Integer, LearnerFeature>();

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

    /**
     * Returns all of the potential values that match the LearnerFeature that featureId belongs to.
     *
     * @param featureId -
     * @return Set
     */
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

    public LearnerFeature getLearnerFeatureFor(int featureId) {
        if (idsToFeatureTypes.containsKey(featureId))
            return idsToFeatureTypes.get(featureId);

        throw new RuntimeException("No valid feature for ID " + featureId); // This will probably be thrown sometime soon
    }

    /**
     * Returns the value for the feature ID provided. So if the feature ID is one that points to "CONTAINS_BIGRAM",
     * it should return a bigram.
     *
     * TODO: Should we include a datastructure to do this for us?
     *
     * @param featureId - ID
     * @return object
     */
    public Object getValueFor(int featureId) {
        Map<Object, Integer> map = ids.get(idsToFeatureTypes.get(featureId));

        for (Map.Entry<Object, Integer> e : map.entrySet()) {
            if (e.getValue() == featureId)
                return e.getKey();
        }

        throw new RuntimeException("Could not find value for ID " + featureId);
    }

    /**
     * Returns the IDs for all of the learner features provided.
     *
     * @param featureTypes -
     * @return Set of int
     */
    public Set<Integer> getIdsFor(Set<LearnerFeature> featureTypes) {
        TreeSet<Integer> set = new TreeSet<Integer>();

        for (LearnerFeature f : featureTypes)
            set.addAll(ids.get(f).values());

        return set;
    }

    /**
     * Save the instance to the disk.
     *
     * TODO: This may need to be redone or something
     *
     * @param learnerFeatureManagerFile -
     */
    public void saveInstance(String learnerFeatureManagerFile) {
        File f = new File(learnerFeatureManagerFile);
        Gson gson = new Gson();

        try {
            PrintWriter writer = new PrintWriter(f);
            writer.print(gson.toJson(this));
            writer.flush();
            writer.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private static LearnerFeatureManager pullFromFile(File f) {
        // TODO: THIS
        return null;
    }
}
