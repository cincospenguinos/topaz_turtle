package toptur;

/**
 * A POJO that represents a word from SentiWordNet
 */
public class SentiWordValues {

    private double positive;
    private double negative;
    private double objectivity;

    public SentiWordValues(double _pos, double _neg) {
        positive = _pos;
        negative = _neg;

        objectivity = 1 - (positive + negative);
    }

    public double getPositivity() {
        return positive;
    }

    public double getNegativity() {
        return negative;
    }

    public double getObjectivity() {
        return objectivity;
    }

    public void includeNewValues(double pos, double neg) {
        positive = (positive + pos) / 2;
        negative = (negative + neg) / 2;
        objectivity = 1 - (positive + negative);
    }
}
