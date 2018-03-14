package toptur;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;

public class LibLinearFeatureManager {
    public enum LibLinearFeature {
        // These are for sentence, targets and agents
        CONTAINS_UNIGRAM, CONTAINS_BIGRAM, OBJECTIVITY_OF_SENTENCE, HAS_WORD_RELATED_TO_OTHER_WORD, OBJECTIVITY_OF_RELATED_WORD, // TODO: Add other features for sentence!

        // These are for extracting the actual opinion words
        PREVIOUS_UNIGRAM, THIS_UNIGRAM, NEXT_UNIGRAM, PREVIOUS_PART_OF_SPEECH, THIS_PART_OF_SPEECH, NEXT_PART_OF_SPEECH, OBJECTIVITY_OF_WORD
    }

    private static volatile LibLinearFeatureManager instance;

    private int counter;
    private HashMap<LibLinearFeature, TreeMap<Object, Integer>> ids;

    private LibLinearFeatureManager() {
        counter = 1;
        ids = new HashMap<LibLinearFeature, TreeMap<Object, Integer>>();

        for (LibLinearFeature f : LibLinearFeature.values())
            ids.put(f, new TreeMap<Object, Integer>());
    }

    public static LibLinearFeatureManager getInstance(String filename) {
        if (instance == null) {
            instance = new LibLinearFeatureManager();

            if (new File(filename).exists()) {
                Gson gson = new Gson();
                try {
                    Scanner s = new Scanner(new File(filename));

                    while(s.hasNextLine()) {
                        LibLinearFeature f = LibLinearFeature.valueOf(s.nextLine().trim());
                        TreeMap<Object, Integer> map = gson.fromJson(s.nextLine(), new TypeToken<TreeMap<Object, Integer>>(){}.getType());

                        instance.ids.put(f, map);
                    }

                    s.close();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
            }
        }

        return instance;
    }

    public static void saveInstance(String filename) {
        Gson gson = new Gson();

        try {
            PrintWriter writer = new PrintWriter(new File(filename));

            for (Map.Entry<LibLinearFeature, TreeMap<Object, Integer>> entry : instance.ids.entrySet()) {
                writer.println(entry.getKey());
                writer.println(gson.toJson(entry.getValue(), new TypeToken<TreeMap<Object, Integer>>(){}.getType()));
            }

            writer.flush();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    /**
     * Returns liblinear ID for some LibLinearFeature.
     *
     * @param feature - Enumerated feature type
     * @param value - The value of the feature
     * @return ID representing that value for the given feature
     */
    public int getIdFor(LibLinearFeature feature, Object value) {
        TreeMap<Object, Integer> map = ids.get(feature);
        if (map.containsKey(value))
            return map.get(value);

        int id = counter;
        map.put(value, counter++);

        return id;
    }
}
