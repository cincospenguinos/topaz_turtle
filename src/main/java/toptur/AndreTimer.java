package toptur;

import java.util.Map;
import java.util.Stack;
import java.util.TreeMap;

/**
 * Timer class to help keep track how much of a speedup we get with everything. Has internal stack to manage
 * separate times.
 */
public class AndreTimer {

    private volatile TreeMap<String, Long> times;
    private volatile Stack<AndreTimerFrame> runningTimes;
    private volatile AndreTimerFrame currentFrame;

    public AndreTimer() {
        times = new TreeMap<String, Long>();
        runningTimes = new Stack<AndreTimerFrame>();
    }

    /**
     * Starts a timer matching the trial name
     * @param trialName - name of trial
     */
    public void start(String trialName) {
        AndreTimerFrame frame = new AndreTimerFrame(trialName);

        if (currentFrame != null)
            runningTimes.push(new AndreTimerFrame(currentFrame));

        currentFrame = frame;
        currentFrame.start();
    }

    /**
     * Stops the current timer.
     */
    public void stop() {
        currentFrame.stop();
        times.put(currentFrame.getName(), currentFrame.totalTimeMillis());

        if (!runningTimes.empty())
            currentFrame = runningTimes.pop();
        else
            currentFrame = null;
    }

    /**
     * Stops the timer matching the name provided. This method is designed to allow threaded testing,
     * however IT DOES SKEW TIMINGS TO BE SLOWER. Use this only if you are worried about threading issues.
     *
     * @param name -
     */
    public void stop(String name) {
        if (currentFrame.getName().equals(name))
            stop();

        AndreTimerFrame frame = null;

        for (AndreTimerFrame f : runningTimes){
            if (f.getName().equals(name)) {
                f.stop();
                times.put(currentFrame.getName(), currentFrame.totalTimeMillis());
                frame = f;
                break;
            }
        }

        if (frame != null)
            runningTimes.remove(frame);
    }

    /**
     * Returns nice printout of all the times that this class has setup.
     * @return String
     */
    public String toString() {
        StringBuilder builder = new StringBuilder();

        for (Map.Entry<String, Long> e : times.entrySet()) {
            builder.append(e.getKey());
            builder.append("\t");
            builder.append(e.getValue() / 1000.0);
            builder.append("s\n");
        }

        return builder.toString();
    }

    public long timeInMillisFor(String name) {
        return times.get(name);
    }
}
