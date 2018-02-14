package toptur;

import edu.stanford.nlp.simple.Document;
import edu.stanford.nlp.simple.Sentence;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;

/**
 * Main class for the project.
 *
 * Here's the usage:
 *
 *     args[0] -- what task we are going to do
 *     args[1+] -- whatever input is needed
 *
 * so
 */
public class Main {

    public static final String DEV_DOCS = "dataset/dev";
    public static final String TEST_DOCS = "dataset/test";
    public static final String ORIGINAL_DOCS = "dataset/original_dataset/docs";
    public static final String SENTENCES_TRAINING_FILE = "sentences_train.vector";
    public static final String SENTENCES_MODEL_FILE = ".liblinear_models/sentences.model";
    public static final String SENTENCES_TEST_FILE = "sentences_test.vector";

    public static void main(String[] args) {
        if (args.length == 0)
            System.exit(0);

        setup();

        // TODO: Goal for today is to be able to detect with some amount of accuracy whether a sentence does or does not have an opinion

        String task = args[0].toLowerCase();
        if (task.equals("train")) {
            ArrayList<NewsArticle> devArticles = getAllDocsFrom(DEV_DOCS);
            ArrayList<NewsArticle> testArticles = getAllDocsFrom(TEST_DOCS);

            // Let's start by creating vector files for liblinear
            createVectorFile(devArticles, SENTENCES_TRAINING_FILE);
            createVectorFile(testArticles, SENTENCES_TEST_FILE);
            trainLibLinear(SENTENCES_TRAINING_FILE, SENTENCES_MODEL_FILE);
            testLibLinear(SENTENCES_TEST_FILE, SENTENCES_MODEL_FILE, "output.txt");

        } else if (task.equals("test")) {
            ArrayList<NewsArticle> testArticles = getAllDocsFrom(TEST_DOCS);

        } else if (task.equals("extract")) {
            for (int i = 1; i < args.length; i++) {
                // TODO: This
                System.out.println(args[i] + " is a file I'll need to extract stuff from");
            }

        } else {
            System.err.println("No valid task provided");
            System.exit(1);
        }
    }

    /**
     * Helper method to get all docs in some path in NewsArticle class.
     *
     * @param path
     * @return
     */
    private static ArrayList<NewsArticle> getAllDocsFrom(String path) {
        ArrayList<NewsArticle> docs = new ArrayList<NewsArticle>();

        File folder = new File(path);

        if (!folder.exists()) {
            System.err.println("Could not find folder \"" + path + "\"!");
            System.exit(1);
        }

        for(File f : folder.listFiles()) {
            docs.add(NewsArticle.fromJson(f));
        }

        return docs;
    }

    /**
     * Generates a vector file in LibLinear format for whatever articles are provided.
     *
     * @param articles
     * @param nameOfVectorFile
     */
    private static void createVectorFile(ArrayList<NewsArticle> articles, String nameOfVectorFile) {
        StringBuilder vectorFileBuilder = new StringBuilder();
        LibLinearFeatureManager libLinearFeatureManager = LibLinearFeatureManager.getInstance();

        for (NewsArticle article : articles) {
            Document doc = new Document(article.getFullText());

            for (Sentence s : doc.sentences()) {
                StringBuilder vectorLineBuilder = new StringBuilder();
                TreeMap<Integer, Boolean> libLinearFeatureVector = new TreeMap<Integer, Boolean>();

                // The label for this sentence
                if (article.sentenceHasOpinion(s.toString()))
                    vectorLineBuilder.append(1);
                else
                    vectorLineBuilder.append(0);

                // Let's just grab the unigrams for now
                for (String w : s.words()) {
                    int id = libLinearFeatureManager.getIdFor(LibLinearFeatureManager.LibLinearFeature.CONTAINS_UNIGRAM, w);
                    libLinearFeatureVector.put(id, true);
                }

                for (Map.Entry<Integer, Boolean> e : libLinearFeatureVector.entrySet()) {
                    vectorLineBuilder.append(" ");
                    vectorLineBuilder.append(e.getKey());
                    vectorLineBuilder.append(":");

                    if (e.getValue())
                        vectorLineBuilder.append(1);
                }

                vectorFileBuilder.append(vectorLineBuilder.toString());
                vectorFileBuilder.append("\n");
            }
        }

        try {
            PrintWriter vectorFile = new PrintWriter(nameOfVectorFile);
            vectorFile.print(vectorFileBuilder.toString());
            vectorFile.flush();
            vectorFile.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private static void trainLibLinear(String vectorFileName, String modelFileName) {
        // Now that the vector file is put together, we need to run liblinear
        try {
            Runtime.getRuntime().exec("./liblinear_train " + vectorFileName + " " + modelFileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void testLibLinear(String testVectorFileName, String modelFileName, String outputFileName) {
        // Now that the vector file is put together, we need to run liblinear
        try {
            System.out.println("Predicting \"" + testVectorFileName + "\" with liblinear...");

            Runtime runtime = Runtime.getRuntime();
            Scanner s = new Scanner(runtime.exec("./liblinear_predict " + testVectorFileName + " " + modelFileName + " " + outputFileName).getInputStream());
            while (s.hasNextLine())
                System.out.println(s.nextLine());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void setup() {

    }
}
