package toptur;

import java.util.List;
import java.util.Set;

public class DecisionTree {

    private static LearnerFeatureManager manager;

    private int featureId;


    public DecisionTree(List<LearnerExample> examples, Set<LearnerFeature> features, int maxDepth) {
        // TODO: all of this

        if (examplesHaveSameLabel(examples) || maxDepth == 0) {

        } else {
            // Find the best feature

            // Add a new branch for that best feature
        }
    }

    public static void setManager(LearnerFeatureManager _manager) {
        manager = _manager;
    }

    /**
     * Returns depth of this tree.
     *
     * @return int
     */
    public int depth() {
        return 0; // TODO: This
    }

    /**
     * Helper method. Returns true if every example has the exact same label.
     *
     * @param examples -
     * @return boolean
     */
    private boolean examplesHaveSameLabel(List<LearnerExample> examples) {
        // TODO: This
        return false;
    }
}
