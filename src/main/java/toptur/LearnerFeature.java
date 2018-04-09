package toptur;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.TreeSet;

/**
 * Represents a LearnerFeature
 *
 * TODO: Add some more features here!
 */
public enum LearnerFeature {
    CONTAINS_UNIGRAM, CONTAINS_BIGRAM, THIS_WORD, THIS_POS, SUBJECTIVITY_OF_WORD;

    private static final boolean[] BOOL_VALS = new boolean[] { true, false };
    private static final int[] BIO_VALS = new int[] { 0, 1, 2 };

    public Set<Object> possibleValuesForFeature() {
        switch(this) {
            case CONTAINS_UNIGRAM:
            case CONTAINS_BIGRAM:
                HashSet<Object> hashSet = new HashSet<Object>();
                for (boolean b : BOOL_VALS)
                    hashSet.add(b);
                return hashSet;
        }

        throw new RuntimeException("A value for a given feature must be defined!");
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
        set.add(LearnerFeature.SUBJECTIVITY_OF_WORD);
        return set;
    }
}
