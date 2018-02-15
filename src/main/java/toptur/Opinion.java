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

    public boolean equals(Object o) {
        if (o instanceof Opinion) {
            Opinion other = (Opinion) o;

            return other.sentence.equals(sentence) && other.opinion.equals(opinion) && other.agent.equals(agent)
                    && other.target.equals(target) && other.sentiment.equals(sentiment);
        }

        return false;
    }
}
