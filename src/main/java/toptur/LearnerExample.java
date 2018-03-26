package toptur;

import com.sun.istack.internal.NotNull;

/**
 * Learner example takes in some example Object E (could be a string, a document, whatever)
 * and a label of some sort (Boolean, Integer, etc.)
 *
 * NOTE: This design offloads the determination of feature values out of this class and to the user
 * of this class, by providing an instance of LearnerExampleValueRequestListener. I don't know if
 * that's a good idea, but it's what I'm doing.
 *
 * @param <E> - Example type
 * @param <L> - Label type
 */
public class LearnerExample<E, L> {
    private E example;
    private L label;
    private LearnerExampleValueRequestListener<E> listener;

    public LearnerExample(E e, L l, @NotNull LearnerExampleValueRequestListener<E> _listener) {
        example = e;
        label = l;
        listener = _listener;
    }

    public E getExample() {
        return example;
    }

    public L getLabel() {
        return label;
    }

    /**
     * Returns true if the feature id provided has a value matching the value provided.
     *
     * @param featureId -
     * @param value -
     * @return true if ^^^^
     */
    public boolean featureMatchesValue(int featureId, Object value) {
        return valueOf(featureId).equals(value);
    }

    /**
     * Returns the value of the feature matching the ID provided.
     *
     * @param featureId -
     * @return Object value
     */
    public Object valueOf(int featureId) {
        return listener.valueOfFeatureForExample(example, featureId);
    }
}
