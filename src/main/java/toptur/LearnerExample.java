package toptur;

/**
 * Learner example takes in some example Object E (could be a string, a document, whatever)
 * and a label of some sort (Boolean, Integer, etc.)
 *
 * @param <E> - Example type
 * @param <L> - Label type
 */
public class LearnerExample<E, L> {
    private E example;
    private L label;

    public LearnerExample(E e, L l) { example = e; label = l; }

    public E getExample() {
        return example;
    }

    public L getLabel() {
        return label;
    }

    public boolean featureMatchesValue(int featureId, Object value) {
        return false; // TODO: This so hard
    }

    public Object valueOf(int featureId) {
        return -1; // TODO: This as well
    }
}
