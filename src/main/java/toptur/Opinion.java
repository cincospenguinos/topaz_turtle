package toptur;

import edu.stanford.nlp.simple.Sentence;

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

            return other.sentence.equals(this.sentence)
                    && opinionsMatch(other.opinion, this.opinion)
//                    && other.agent.equals(this.agent)
//                    && other.target.equals(this.target)
//                    && other.sentiment.equals(this.sentiment)
                    ;

        }

        return false;
    }

    private boolean opinionsMatch(String o1, String o2) {
        return o1.contains(o2) || o2.contains(o1);
    }
}
