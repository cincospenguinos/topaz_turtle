package toptur;

import edu.stanford.nlp.simple.Document;
import edu.stanford.nlp.simple.Sentence;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

/**
 * Main class for the project.
 *
 * Here's the usage:
 *
 *     args[0] -- what task we are going to do
 *     args[1+] -- whatever input is needed
 */
public class Main {

    public static final String DEV_DOCS = "dataset/dev";
    public static final String TEST_DOCS = "dataset/test";
    public static final String ORIGINAL_DOCS = "dataset/original_dataset/docs";
    public static final String SENTENCES_TRAINING_FILE = "sentences_train.vector";
    public static final String SENTENCES_MODEL_FILE = ".liblinear_models/sentences.model";
    public static final String SENTENCES_TEST_FILE = "sentences_test.vector";
    public static final String SENTI_WORD_NET_FILE = "sentiwordnet.txt";

    private static SentiWordNetDictionary sentiWordNetDictionary;

    // TODO: Next goal: extract the specific opinion word(s) from the sentence

    public static void main(String[] args) {
        if (args.length == 0)
            System.exit(0);

        setup();

        System.out.println("Gathering SentiWordNet dictionary...");
        getSentiWordNet();

        String task = args[0].toLowerCase();
        if (task.equals("train")) {
            ArrayList<NewsArticle> devArticles = getAllDocsFrom(DEV_DOCS);
            ArrayList<NewsArticle> testArticles = getAllDocsFrom(TEST_DOCS);

            // Train to detect opinionated sentences
            createVectorFile(devArticles, SENTENCES_TRAINING_FILE);
            createVectorFile(testArticles, SENTENCES_TEST_FILE);
            trainLibLinear(SENTENCES_TRAINING_FILE, SENTENCES_MODEL_FILE);
            testLibLinear(SENTENCES_TEST_FILE, SENTENCES_MODEL_FILE, "/dev/null");

//            for (NewsArticle a : devArticles) {
//                for (Opinion o : a.getGoldStandardOpinions())
//                    System.out.println(o.opinion + "\t" + o.sentence);
//            }

        } else if (task.equals("test")) {
            ArrayList<NewsArticle> testArticles = getAllDocsFrom(TEST_DOCS);
//            createVectorFile(testArticles, SENTENCES_TEST_FILE);
            testLibLinear(SENTENCES_TEST_FILE, SENTENCES_MODEL_FILE, "/dev/null");

            // Extract the opinions
            for (NewsArticle a : testArticles) {
                Document doc = new Document(a.getFullText());

                for (Sentence s : doc.sentences()) {
                    if (sentenceContainsOpinion(s)) {
                        Opinion o = new Opinion();
                        o.sentence = s.toString();

                        // Grab the opinion words
                        Sentence sent = new Sentence(o.sentence);

                        double objectivity = 1.0;
                        String mostSubjective = null;

                        for (String w : sent.words()) {
                            if (sentiWordNetDictionary)
                        }

                        // TODO: Grab the opinion/target/agent/etc. from the sentence

                        a.addExtractedOpinion(o);
                    }
                }


            }

            evaluateExtractedOpinions(testArticles);

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
                TreeMap<Integer, Object> libLinearFeatureVector = new TreeMap<Integer, Object>();

                // The label for this sentence
                if (article.sentenceHasOpinion(s.toString()))
                    vectorLineBuilder.append(1);
                else
                    vectorLineBuilder.append(0);

                List<String> words = s.words();

                // Creating the feature vectors
                for (LibLinearFeatureManager.LibLinearFeature feature : LibLinearFeatureManager.LibLinearFeature.values()) {
                    int id;

                    switch(feature) {
                        case CONTAINS_UNIGRAM:
                            for (String w : words) {
                                id = libLinearFeatureManager.getIdFor(feature, w);
                                libLinearFeatureVector.put(id, true);
                            }
                            break;
                        case OBJECTIVITY_OF_SENTENCE:
                            id = libLinearFeatureManager.getIdFor(feature, true);
                            double objectivity = 0.0;

                            for (String w : words) {
                                objectivity += sentiWordNetDictionary.getObjectivityOf(w);
                            }

                            objectivity /= words.size();
                            libLinearFeatureVector.put(id, objectivity);

                            break;

//                        case HAS_WORD_RELATED_TO_OTHER_WORD:
//                            for (String w : words) {
//                                DataMuseWord[] dataMuseWords = DataMuse.getWordsRelatedTo(w);
//                                if (dataMuseWords == null)
//                                    continue;
//
//                                for (DataMuseWord dmw : dataMuseWords) {
//                                    id = libLinearFeatureManager.getIdFor(feature, dmw.word);
//                                    libLinearFeatureVector.put(id, true);
//                                }
//                            }
//                            break;
                    }
                }

                for (Map.Entry<Integer, Object> e : libLinearFeatureVector.entrySet()) {
                    vectorLineBuilder.append(" ");
                    vectorLineBuilder.append(e.getKey());
                    vectorLineBuilder.append(":");

                    if (e.getValue() instanceof Boolean)
                        vectorLineBuilder.append(1);
                    else
                        vectorLineBuilder.append(e.getValue());
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

    private static void createVectorFile(Sentence sentence, String nameOfVectorFile) {
        StringBuilder vectorFileBuilder = new StringBuilder();
        LibLinearFeatureManager libLinearFeatureManager = LibLinearFeatureManager.getInstance();

        StringBuilder vectorLineBuilder = new StringBuilder();
        TreeMap<Integer, Object> libLinearFeatureVector = new TreeMap<Integer, Object>();

        vectorFileBuilder.append("0"); // Assume objective

        List<String> words = sentence.words();

        // Creating the feature vectors
        for (LibLinearFeatureManager.LibLinearFeature feature : LibLinearFeatureManager.LibLinearFeature.values()) {
            switch(feature) {
                case CONTAINS_UNIGRAM:
                    for (String w : words) {
                        int id = libLinearFeatureManager.getIdFor(feature, w);
                        libLinearFeatureVector.put(id, true);
                    }
                    break;
                case OBJECTIVITY_OF_SENTENCE:
                    int id = libLinearFeatureManager.getIdFor(feature, true);
                    double objectivity = 0.0;

                    for (String w : words) {
                        objectivity += sentiWordNetDictionary.getObjectivityOf(w);
                    }

                    objectivity /= words.size();
                    libLinearFeatureVector.put(id, objectivity);

                    break;
            }
        }

        for (Map.Entry<Integer, Object> e : libLinearFeatureVector.entrySet()) {
            vectorLineBuilder.append(" ");
            vectorLineBuilder.append(e.getKey());
            vectorLineBuilder.append(":");

            if (e.getValue() instanceof Boolean)
                vectorLineBuilder.append(1);
            else
                vectorLineBuilder.append(e.getValue());
        }

        vectorFileBuilder.append(vectorLineBuilder.toString());
        vectorFileBuilder.append("\n");

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

    private static void setup() {}

    private static void getSentiWordNet() {
        sentiWordNetDictionary = new SentiWordNetDictionary();

        try {
            Scanner s = new Scanner(new File(SENTI_WORD_NET_FILE));
            while(s.hasNextLine()) {
                String line = s.nextLine();

                if (line.startsWith("#"))
                    continue;

                String[] lineParts = line.split("\t");

                if (lineParts.length != 6) {
                    System.out.println(line);
                    throw new RuntimeException("One of the lines in the SENTI_WORD_NET_FILE was not formatted properly!");
                }

                double positive = Double.parseDouble(lineParts[2]);
                double negative = Double.parseDouble(lineParts[3]);

                String[] words = lineParts[4].split(" ");
                for (String w : words) {
                    sentiWordNetDictionary.addWord(w.split("#")[0].replaceAll("_", " "), positive, negative);
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private static boolean sentenceContainsOpinion(Sentence sentence) {
        String name = "some_file.vector";
        createVectorFile(sentence, name);

        try {
            Runtime.getRuntime().exec("./liblinear_predict " + name + " " + SENTENCES_MODEL_FILE + " output.txt");
            Thread.sleep(10); // To give LibLinear enough time to output to file
            Scanner derp = new Scanner(Runtime.getRuntime().exec("cat output.txt").getInputStream());
            int i = derp.nextInt();
            derp.close();
            return i == 1;
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return false;
    }

    private static void evaluateExtractedOpinions(ArrayList<NewsArticle> articles) {
        // So we're using F Score. That means we care about two things:
        // Precision: How many did I extract were correct?
        // Recall: How many correct ones were successfully extracted?

        double totalFScore = 0.0;

        System.out.println("Name\tPrecision\tRecall\tFScore");

        for (NewsArticle article : articles) {
            ArrayList<Opinion> extractedOpinions = (ArrayList<Opinion>) article.getExtractedOpinions().clone();
            ArrayList<Opinion> goldStandardOpinions = (ArrayList<Opinion>) article.getGoldStandardOpinions().clone();

            TreeSet<Integer> extractedIndexes = new TreeSet<Integer>();
            TreeSet<Integer> goldIndexes = new TreeSet<Integer>();

            double truePositives = 0.0;

            // Let's only evaluate how well we discovered sentences
            for (int i = 0; i < goldStandardOpinions.size(); i++) {
                for (int j = 0; j < extractedOpinions.size(); j++) {
                    Opinion gold = goldStandardOpinions.get(i);
                    Opinion extracted = extractedOpinions.get(j);

                    // TODO: Change this to evaluate everything--sentence, opinion, agent, and target
                    if (gold.sentence.equals(extracted.sentence) && !extractedIndexes.contains(j) && !goldIndexes.contains(i)) {
                        truePositives += 1;
                        extractedIndexes.add(j);
                        goldIndexes.add(i);
                    }
                }
            }

//            System.out.println(truePositives + "\t" + goldStandardOpinions.size() + "\t" + extractedOpinions.size());

            double precision = truePositives / extractedOpinions.size();
            double recall = truePositives / goldStandardOpinions.size();
            double fscore = 2 * ((precision * recall) / (precision + recall));

            System.out.println(article.getDocumentName() + "\t" + precision + "\t" + recall + "\t" + fscore);
            totalFScore += fscore;
        }

        System.out.println("\nTOTAL FSCORE: " + totalFScore / articles.size());
    }
}
