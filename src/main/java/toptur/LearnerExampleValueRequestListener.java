package toptur;

/**
 * An interface that makes the user of LearnerExample provide a means of determining the value of a given ID.
 */
public interface LearnerExampleValueRequestListener<E> {
    public Object valueOfFeatureForExample(E example, int featureId);
}
