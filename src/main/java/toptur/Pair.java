package toptur;

public class Pair<A, B> {

    private A a;
    private B b;

    public Pair(A _a, B _b) {
        a = _a;
        b = _b;
    }

    public A getFirst() {
        return a;
    }

    public B getSecond() {
        return b;
    }
}
