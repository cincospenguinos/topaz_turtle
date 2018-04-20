package toptur;

import java.util.*;
import java.util.concurrent.*;

/**
 * Okay, here's how this is going to work:
 *
 * - Every feature for a learner has a unique unsigned integer ID
 * - The values of these features depend completely on the corpus
 *
 * TODO: This needs to be tested soooooo bad
 *
 * @param <E> Example type (a String, a sentence, a document)
 * @param <L> Label type (a boolean value, an integer, a double)
 */
public class DecisionTree<E, L> {

    private int featureId;
    private Map<Object, DecisionTree<E, L>> children;
    private L label;

    public DecisionTree(List<LearnerExample<E, L>> examples, Set<Integer> features, int maxDepth) {
        if (examplesHaveSameLabel(examples) || maxDepth == 0) {
            label = examples.get(0).getLabel();
        } else {
            featureId = bestFeatureFor(examples, features); // Find the best feature

            // Add a new branch for that best feature
            children = new TreeMap<Object, DecisionTree<E, L>>();
            for (Object v : LearnerFeatureManager.getInstance(Main.LEARNER_FEATURE_MANAGER_FILE).getLearnerFeatureFor(featureId).possibleValuesForFeature()) {
                List<LearnerExample<E, L>> subset = getSubsetWhereFeatureEqualsValue(examples, featureId, v);

                if (subset.size() == 0) {
                    children.put(v, new DecisionTree<E, L>(getMajorityLabel(examples)));
                } else {
                    Set<Integer> subFeatures = new TreeSet<Integer>(features);
                    subFeatures.remove(featureId);
                    children.put(v, new DecisionTree<E, L>(subset, subFeatures, maxDepth - 1));
                }
            }
        }
    }

    private DecisionTree(L _label) {
        label = _label;
    }

    /**
     * Returns a guess for a given learner example.
     *
     * @param example - Example to guess for
     * @return <L>
     */
    public L guessFor(LearnerExample<E, L> example) {
        if (label != null)
            return label;

        Object featureValue = example.valueOf(featureId);
        DecisionTree<E, L> child = children.get(featureValue);

        if (child == null) { // If child is null, try converting to string
            child = children.get(featureValue.toString());
        }

        if (child == null) { // If child is STILL null, then just pick the first child
            child = children.get(children.keySet().toArray()[0]);
        }

        return child.guessFor(example);
    }

    /**
     * Returns depth of this tree.
     *
     * @return int
     */
    public int depth() {
        if (children == null)
            return 0;

        int max = -1;
        for (DecisionTree d : children.values())
            max = Math.max(max, d.depth());

        return max + 1;
    }

    /**
     * Helper method. Returns true if every example has the exact same label.
     *
     * @param examples -
     * @return boolean
     */
    private boolean examplesHaveSameLabel(List<LearnerExample<E, L>> examples) {
        Set<L> set = new HashSet<L>();

        for (LearnerExample<E, L> e : examples)
            set.add(e.getLabel());

        return set.size() == 1;
    }

    /**
     * Helper method. Returns best feature given some set of examples and features to describe those examples.
     * Right now it uses majority error--that is, Information Gain = MajErr(S) - SUM_v:Values(A)(|Sv| / |S| MajErr(Sv))
     *
     * @param examples -
     * @param features -
     * @return int
     */
    private int bestFeatureFor(final List<LearnerExample<E, L>> examples, Set<Integer> features) {
        double bestError = -1.0;
        int bestFeature = -1;

        double overallMajorityError = majorityError(examples);

        // First we are going to setup a map with the different features and potential values and stuff
        // TODO: Figure out how much of a speedup we get from multithreading this
        Map<Integer, Future<Double>> map = new TreeMap<Integer, Future<Double>>();
        ExecutorService pool = Executors.newFixedThreadPool(4);

        // Now we will discover features by throwing them in a thread pool
        for (final int id : features) {
            final Set<Object> potentialValues = LearnerFeatureManager.getInstance(Main.LEARNER_FEATURE_MANAGER_FILE).getLearnerFeatureFor(id).possibleValuesForFeature();

            Callable<Double> action = new Callable<Double>() {
                public Double call() throws Exception {
                    double majErr = 0.0;
                    for (Object v : potentialValues) {
                        List<LearnerExample<E, L>> subset = getSubsetWhereFeatureEqualsValue(examples, id, v);
                        majErr += majorityError(subset) * ((double) subset.size() / (double) examples.size());
                    }

                    return majErr;
                }
            };

            Future<Double> f = pool.submit(action);
            map.put(id, f);
        }

        // Finally we will Loop through them all and find the best one
        for (Map.Entry<Integer, Future<Double>> e : map.entrySet()) {
            double majErr;
            try {
                majErr = e.getValue().get();

                if (bestError < overallMajorityError - majErr) {
                    bestError = overallMajorityError - majErr;
                    bestFeature = e.getKey();
                }
            } catch (InterruptedException e1) {
                e1.printStackTrace();
            } catch (ExecutionException e1) {
                e1.printStackTrace();
            }
        }

        pool.shutdownNow();

        if (bestFeature == -1) {
            return (Integer) features.toArray()[0];
        }

        return bestFeature;
    }

    /**
     * Returns majority error for some set of examples.
     *
     * @param examples -
     * @return double
     */
    private double majorityError(List<LearnerExample<E, L>> examples) {
        // First count up each of the labels
        Map<L, Double> counts = new TreeMap<L, Double>();

        for (LearnerExample<E, L> e : examples) {
            if (counts.containsKey(e.getLabel()))
                counts.put(e.getLabel(), counts.get(e.getLabel()) + 1.0);
            else
                counts.put(e.getLabel(), 1.0);
        }

        // Find the majority label
        double highestCount = -1.0;
        L majorityLabel = null;
        for (Map.Entry<L, Double> e : counts.entrySet()) {
            if (e.getValue() > highestCount) {
                highestCount = e.getValue();
                majorityLabel = e.getKey();
            }
        }

        // Now find how many incorrect guesses we would have if we used the majority label
        double wrongGuesses = 0.0;
        for (Map.Entry<L, Double> e : counts.entrySet()) {
            if (!e.getKey().equals(majorityLabel))
                wrongGuesses += e.getValue();
        }

        // MajError = wrong guesses / total examples
        return wrongGuesses / (double) examples.size();
    }

    /**
     * Get all examples that contains some feature matching the value provided.
     *
     * @param examples -
     * @param featureId -
     * @param value -
     * @return List
     */
    private List<LearnerExample<E, L>> getSubsetWhereFeatureEqualsValue(List<LearnerExample<E, L>> examples, int featureId, Object value) {
        List<LearnerExample<E, L>> subset = new ArrayList<LearnerExample<E, L>>();

        for (LearnerExample<E, L> e : examples) {
            if (e.featureMatchesValue(featureId, value)) {
                subset.add(e);
            }
        }

        return subset;
    }

    /**
     * Helper method. Returns the majority label given some list of examples.
     *
     * @param examples -
     * @return <L>
     */
    private L getMajorityLabel(List<LearnerExample<E, L>> examples) {
        if (examples.size() == 0)
            throw new RuntimeException("Cannot get majority label of empty list of examples.");

        Map<L, Integer> counts = new TreeMap<L, Integer>();

        for (LearnerExample<E, L> e : examples) {
            if (counts.containsKey(e.getLabel()))
                counts.put(e.getLabel(), counts.get(e.getLabel()) + 1);
            else
                counts.put(e.getLabel(), 1);
        }

        L majorityLabel = null;
        int highestCount = -1;

        for (Map.Entry<L, Integer> e : counts.entrySet()) {
            if (e.getValue() > highestCount) {
                highestCount = e.getValue();
                majorityLabel = e.getKey();
            }
        }

        return majorityLabel;
    }
}
