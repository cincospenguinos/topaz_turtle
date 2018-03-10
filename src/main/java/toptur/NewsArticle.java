package toptur;

import com.google.gson.Gson;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;

/**
 * Represents a single document we'll extract information from.
 */
public class NewsArticle {

    private String documentName;
    private String fullText;

    // Maps opinion expression to opinion object
    private HashMap<String, Opinion> extractedOpinions;
    private HashMap<String, Opinion> goldStandardOpinions;

    public NewsArticle(File f) {
        documentName = f.getName();

        try {
            StringBuilder fullTextBuilder = new StringBuilder();
            Scanner s = new Scanner(f);

            while(s.hasNextLine()) {
                fullTextBuilder.append(s.nextLine());
                fullTextBuilder.append("\n");
            }

            s.close();

            fullText = fullTextBuilder.toString();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(1);
        }

        extractedOpinions = new HashMap<String, Opinion>();
        goldStandardOpinions = new HashMap<String, Opinion>();
    }

    private NewsArticle(String name, String _fullText, Opinion[] goldStandards) {
        documentName = name;
        fullText = _fullText;
        extractedOpinions = new HashMap<String, Opinion>();
        goldStandardOpinions = new HashMap<String, Opinion>();
        
        for (Opinion o : goldStandards) {
        		goldStandardOpinions.put(o.opinion, o);
        }
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("# ");
        builder.append(documentName);
        builder.append("\n\n");

        for (String o : extractedOpinions.keySet()) {
            builder.append("\tOpinion\t");
            builder.append(extractedOpinions.get(o).opinion);
            builder.append("\n\tAgent\t");
            builder.append(extractedOpinions.get(o).agent);
            builder.append("\n\tTarget\t");
            builder.append(extractedOpinions.get(o).target);
            builder.append("\n\tSentiment\t");
            builder.append(extractedOpinions.get(o).sentiment);
            builder.append("\n\tSentence\t");
            builder.append(extractedOpinions.get(o).sentence);
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

        return new NewsArticle(jsonFile.getName(), fullTextBuilder.toString(), opinions);
    }

    /**
     * Checks if the sentence passed EXACTLY MATCHES a sentence with an opinion, as
     * found in the gold standard opinion set.
     *
     * @param sentence
     * @return
     */
    public boolean sentenceHasOpinion(String sentence) {
//        for (Opinion o : goldStandardOpinions)
//            if (o.sentence.equalsIgnoreCase(sentence))
//                return true;
    		if(goldStandardOpinions.containsKey(sentence))
    			return true;
        return false;
    }
    
    public String getOpinionAgent(String sentence) {
//    		for (Opinion o: goldStandardOpinions)
//    			if (o.sentence.equalsIgnoreCase(sentence))
//    				return o.agent;
	    	if(goldStandardOpinions.containsKey(sentence))
				return goldStandardOpinions.get(sentence).agent;
    		return "Null";
    }

    public String getFullText() {
        return fullText;
    }

    public HashMap<String, Opinion> getGoldStandardOpinions() {
        return goldStandardOpinions;
    }

    public void addExtractedOpinion(Opinion o) {
        extractedOpinions.put(o.sentence, o);
    }

    public HashMap<String, Opinion> getExtractedOpinions() {
        return extractedOpinions;
    }

    public String getDocumentName() {
        return documentName;
    }
}
