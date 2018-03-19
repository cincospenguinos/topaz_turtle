package toptur;

public abstract class LearnerExample<E, L> {
    private E example;
    private L learner;

    public LearnerExample(E e, L l) { example = e; learner = l; }

    public E getExample() {
        return example;
    }

    public L getLearner() {
        return learner;
    }
}
