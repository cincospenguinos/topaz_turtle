package toptur;

/**
 * Represents a single frame for the AndreTimer. Kind of a POJO.
 */
public class AndreTimerFrame {

    private String name;
    private long startTimeMillis;
    private long endTimeMillis;

    public AndreTimerFrame(String _name) {
        endTimeMillis = -1;
        name = _name;
    }

    public AndreTimerFrame(AndreTimerFrame copy) {
        name = copy.name;
        startTimeMillis = copy.startTimeMillis;
        endTimeMillis = copy.endTimeMillis;
    }

    public void start() {
        startTimeMillis = System.currentTimeMillis();
    }

    public void stop() {
        endTimeMillis = System.currentTimeMillis();
    }

    public long totalTimeMillis() {
        if (isRunning())
            return -1;

        return endTimeMillis - startTimeMillis;
    }

    public boolean isRunning() {
        return endTimeMillis < 0;
    }

    public String getName() {
        return name;
    }
}
