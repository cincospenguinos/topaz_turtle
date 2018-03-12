package toptur;

/**
 * An opinion, extracted from an article.
 */
public class Opinion {
    public String opinion;
    public String agent;
    public String sentiment;
    public String target;
    public String sentence;

    public static boolean opinionsMatchGivenOption(Opinion o1, Opinion o2, EvaluationOption option) {
        switch(option) {
            case SENTENCE:
                return o1.sentence.equals(o2.sentence);
            case OPINION:
                return opinionExpressionsMatch(o1.opinion, o2.opinion);
            case AGENT:
                return o1.agent.equals(o2.agent);
            case TARGET:
                return o1.target.equals(o2.target);
            case SENTIMENT:
                return o1.sentiment.equals(o2.sentiment);
            default:
                return false;
        }
    }

    public int sentimentId() {
        if (sentiment.equals("neutral"))
            return 0;
        if (sentiment.equals("positive"))
            return 1;
        if (sentiment.equals("negative"))
            return 2;
        if (sentiment.equals("both"))
            return 3;
        if (sentiment.equals("none"))
            return 4;

        throw new RuntimeException(sentiment + " was the sentiment.");
    }

    public static String fromSentimentId(int id) {
        switch(id) {
            case 0:
                return "neutral";
            case 1:
                return "positive";
            case 2:
                return "negative";
            case 3:
                return "both";
            case 4:
                return "none";
        }

        throw new RuntimeException(id + " was the sentiment.");
    }

    private static boolean opinionExpressionsMatch(String o1, String o2) {
        return o1.contains(o2) || o2.contains(o1);
    }
}
