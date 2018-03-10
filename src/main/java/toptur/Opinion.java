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

    private static boolean opinionExpressionsMatch(String o1, String o2) {
        return o1.contains(o2) || o2.contains(o1);
    }
}
