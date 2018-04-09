package toptur;

import java.util.HashSet;
import java.util.Set;
import java.util.TreeSet;

/**
 * Represents a LearnerFeature
 *
 * TODO: Add some more features here!
 */
public enum LearnerFeature {
    CONTAINS_UNIGRAM, CONTAINS_BIGRAM, THIS_WORD, THIS_POS, OBJECTIVITY_OF_WORD;

    private static final boolean[] BOOL_VALS = new boolean[] { true, false };
    private static final int[] BIO_VALS = new int[] { 0, 1, 2 };

    public Set<Object> possibleValuesForFeature() {
        HashSet<Object> set = new HashSet<Object>();

        switch(this) {
            case CONTAINS_UNIGRAM:
            case CONTAINS_BIGRAM:
                for (boolean b : BOOL_VALS)
                    set.add(b);
                return set;
            case THIS_WORD:
                set.addAll(Main.getAllWords());
                return set;
            case THIS_POS:
                set.addAll(Main.getAllPos());
                break;
            case OBJECTIVITY_OF_WORD:
                for (int i = 0; i <= 100; i++) set.add(i);
                break;
            default:
                throw new RuntimeException("A value for a given feature must be defined!");
        }

        return set;
    }

    public static Set<LearnerFeature> getSentenceFeatures() {
        Set<LearnerFeature> set = new TreeSet<LearnerFeature>();
        set.add(LearnerFeature.CONTAINS_UNIGRAM);
        set.add(LearnerFeature.CONTAINS_BIGRAM);
        return set;
    }

    public static Set<LearnerFeature> getOpinionPhraseFeatures() {
        Set<LearnerFeature> set = new TreeSet<LearnerFeature>();
        set.add(LearnerFeature.THIS_WORD);
        set.add(LearnerFeature.THIS_POS);
        set.add(LearnerFeature.OBJECTIVITY_OF_WORD);
        return set;
    }
}
