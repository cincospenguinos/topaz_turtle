package toptur;

import org.threadly.concurrent.UnfairExecutor;

import java.util.*;

/**
 * An implementation of bagged trees, following the same pattern of design as the other learner classes.
 *
 * @param <E>
 * @param <L>
 */
public class BaggedTrees<E, L> {

    private List<DecisionTree<E, L>> trees;

    private static final int EXAMPLE_SUBSET_SIZE = 100;
    private static final int FEATURE_SUBSET_SIZE = 100;

    private static volatile Random random = new Random(1992);

    /**
     * Serial version of generating trees.
     *
     * @param examples -
     * @param featureIds -
     * @param numberOfTrees -
     * @param treeDepth -
     */
    public BaggedTrees(List<LearnerExample<E, L>> examples, Set<Integer> featureIds, int numberOfTrees, int treeDepth) {
        trees = new ArrayList<DecisionTree<E, L>>();

        for (int i = 0; i < numberOfTrees; i++) {
            Set<Integer> featureSubset = new TreeSet<Integer>();
            List<LearnerExample<E, L>> exampleSubset = new ArrayList<LearnerExample<E, L>>();

            for (int j = 0; j < EXAMPLE_SUBSET_SIZE; j++)
                exampleSubset.add(examples.get(random.nextInt(examples.size())));

            Integer[] featureClone = featureIds.toArray(new Integer[0]);
            for (int j = 0; j < FEATURE_SUBSET_SIZE; j++)
                featureSubset.add(featureClone[random.nextInt(featureClone.length)]);

            trees.add(new DecisionTree<E, L>(exampleSubset, featureSubset, treeDepth));
        }
    }

    /**
     * Threaded version of BaggedTrees.
     * @param examples -
     * @param featureIds -
     * @param numberOfTrees -
     * @param treeDepth -
     * @param threads -
     */
    public BaggedTrees(final List<LearnerExample<E, L>> examples, final Set<Integer> featureIds, int numberOfTrees, final int treeDepth, int threads) {
        trees = new ArrayList<DecisionTree<E, L>>();

        UnfairExecutor executor = new UnfairExecutor(threads);

        for (int i = 0; i < numberOfTrees - 1; i++) {
            Runnable runnable = new Runnable() {
                public void run() {
                    Set<Integer> featureSubset = new TreeSet<Integer>();
                    List<LearnerExample<E, L>> exampleSubset = new ArrayList<LearnerExample<E, L>>();

                    for (int j = 0; j < EXAMPLE_SUBSET_SIZE; j++)
                        exampleSubset.add(examples.get(random.nextInt(examples.size())));

                    Integer[] featureClone = featureIds.toArray(new Integer[0]);
                    for (int j = 0; j < FEATURE_SUBSET_SIZE; j++)
                        featureSubset.add(featureClone[random.nextInt(featureClone.length)]);

                    trees.add(new DecisionTree<E, L>(exampleSubset, featureSubset, treeDepth));
                }
            };

            executor.execute(runnable);
        }

        // So that the main thread is doing some work as well
        Set<Integer> featureSubset = new TreeSet<Integer>();
        List<LearnerExample<E, L>> exampleSubset = new ArrayList<LearnerExample<E, L>>();

        for (int j = 0; j < EXAMPLE_SUBSET_SIZE; j++)
            exampleSubset.add(examples.get(random.nextInt(examples.size())));

        Integer[] featureClone = featureIds.toArray(new Integer[0]);
        for (int j = 0; j < FEATURE_SUBSET_SIZE; j++)
            featureSubset.add(featureClone[random.nextInt(featureClone.length)]);

        trees.add(new DecisionTree<E, L>(exampleSubset, featureSubset, treeDepth));

        try {
            executor.awaitTermination();
        } catch (InterruptedException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Submits a guess label for the example provided.
     *
     * @param example -
     * @return L
     */
    public L guessFor(LearnerExample<E, L> example) {
        Map<L, Integer> counts = new TreeMap<L, Integer>();
        for (DecisionTree<E, L> d : trees) {
            L label = d.guessFor(example);

            if (counts.containsKey(label))
                counts.put(label, counts.get(label) + 1);
            else
                counts.put(label, 1);
        }

        int highest = -1;
        L bestLabel = null;

        for (Map.Entry<L, Integer> e : counts.entrySet())
            if (e.getValue() > highest) {
                bestLabel = e.getKey();
                highest = e.getValue();
            }

        return bestLabel;
    }

    /**
     * Returns the guessed label for each tree in the forest.
     *
     * @param example -
     * @return List of L
     */
    public List<L> allGuessesFor(LearnerExample<E, L> example) {
        List<L> labels = new ArrayList<L>();

        for (DecisionTree<E, L> d : trees)
            labels.add(d.guessFor(example));

        return labels;
    }

    /**
     * Set the seed for the randomizer that decides what featureIds are handed to the decision trees.
     *
     * @param seed -
     */
    public static void setSeed(int seed) {
        random = new Random(seed);
    }
}