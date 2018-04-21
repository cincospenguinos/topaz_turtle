package toptur;

import com.google.gson.Gson;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

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
    private static int numThreads = 3;

    /**
     * Serial version of generating trees.
     *
     * @param examples -
     * @param featureIds -
     * @param numberOfTrees -
     * @param treeDepth -
     */
    public BaggedTrees(final List<LearnerExample<E, L>> examples, final Set<Integer> featureIds, int numberOfTrees, final int treeDepth) {
        trees = new ArrayList<DecisionTree<E, L>>();
        ExecutorService pool = Executors.newFixedThreadPool(numThreads);

        for (int i = 0; i < numberOfTrees; i++) {// TODO: Set this up to have the main thread handle some
            pool.submit(new Runnable() {
                public void run() {
                    addTree(examples, featureIds, treeDepth);
                }
            });
        }

        pool.shutdown(); // This will ensure that all the previous threads will execute

        try {
            boolean successful = pool.awaitTermination(10, TimeUnit.MINUTES);

            if (!successful) {
                System.out.println("Not successful! BAGGEDTREES");
                System.exit(1);
            }
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
     * Saves the classifier to the file provided.
     *
     * @param fileName - name of file
     */
    public void saveToFile(String fileName) {
        String json = new Gson().toJson(this);

        try {
            PrintWriter writer = new PrintWriter(new File(fileName));
            writer.print(json);
            writer.flush();
            writer.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static void setNumThreads(int number) {
        numThreads = number;
    }

    /**
     * Set the seed for the randomizer that decides what featureIds are handed to the decision trees.
     *
     * @param seed -
     */
    public static void setSeed(int seed) {
        random = new Random(seed);
    }

    private void addTree(List<LearnerExample<E, L>> examples, Set<Integer> featureIds, int treeDepth) {
        Set<Integer> featureSubset = new TreeSet<Integer>();
        List<LearnerExample<E, L>> exampleSubset = new ArrayList<LearnerExample<E, L>>();

        for (int j = 0; j < EXAMPLE_SUBSET_SIZE; j++)
            exampleSubset.add(examples.get(random.nextInt(examples.size())));

        Integer[] featureClone = featureIds.toArray(new Integer[0]);
        for (int j = 0; j < FEATURE_SUBSET_SIZE; j++)
            featureSubset.add(featureClone[random.nextInt(featureClone.length)]);

        DecisionTree<E, L> tree = new DecisionTree<E, L>(exampleSubset, featureSubset, treeDepth);
        trees.add(tree);
    }
}
