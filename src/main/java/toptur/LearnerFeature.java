package toptur;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public enum LearnerFeature {
    CONTAINS_UNIGRAM, CONTAINS_BIGRAM, PREVIOUS_WORD, THIS_WORD, NEXT_WORD, PREVIOUS_POS, THIS_POS, NEXT_POS;

    private static final boolean[] BOOL_VALS = new boolean[] { true, false };

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
}
