package toptur;

import com.google.gson.Gson;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Represents a single document we'll extract information from.
 */
public class NewsArticle {

    private String documentName;
    private String fullText;
    private ArrayList<Opinion> extractedOpinions; // opinions found in the document
    private ArrayList<Opinion> goldStandardOpinions; // hand annotated opinions

    private NewsArticle(String name, String _fullText, Opinion[] goldStandards) {
        documentName = name;
        fullText = _fullText;
        extractedOpinions = new ArrayList<Opinion>();
        goldStandardOpinions = new ArrayList<Opinion>();

        goldStandardOpinions.addAll(Arrays.asList(goldStandards));
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("# ");
        builder.append(documentName);
        builder.append("\n\n");

        for (Opinion o : goldStandardOpinions) {
            builder.append("\tOpinion\t");
            builder.append(o.opinion);
            builder.append("\n\tAgent\t");
            builder.append(o.agent);
            builder.append("\n\tTarget\t");
            builder.append(o.target);
            builder.append("\n\tSentiment\t");
            builder.append(o.sentiment);
            builder.append("\n\tSentence\t");
            builder.append(o.sentence);
            builder.append("\n\n");
        }

        return builder.toString();
    }

    public static NewsArticle fromJson(File jsonFile) {

        // First grab the gold standard opinions
        Gson gson = new Gson();
        StringBuilder jsonText = new StringBuilder();
        try {
            Scanner s = new Scanner(jsonFile);
            while(s.hasNextLine())
                jsonText.append(s.nextLine());
            s.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        Opinion[] opinions = gson.fromJson(jsonText.toString(), Opinion[].class);

        // Now grab the full text
        StringBuilder fullTextBuilder = new StringBuilder();
        for (File dir : new File(Main.ORIGINAL_DOCS).listFiles()) {
            for (File f : dir.listFiles()) {
                if (f.getName().equals(jsonFile.getName())) {
                    try {
                        Scanner s = new Scanner(f);
                        while(s.hasNextLine())
                            fullTextBuilder.append(s.nextLine());
                        s.close();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                }
            }
        }

        return new NewsArticle(jsonFile.getName(), fullTextBuilder.toString(), opinions); // TODO: This
    }

    /**
     * Checks if the sentence passed EXACTLY MATCHES a sentence with an opinion, as
     * found in the gold standard opinion set.
     *
     * @param sentence
     * @return
     */
    public boolean sentenceHasOpinion(String sentence) {
        for (Opinion o : goldStandardOpinions)
            if (o.sentence.equalsIgnoreCase(sentence))
                return true;

        return false;
    }

    public String getFullText() {
        return fullText;
    }
}
