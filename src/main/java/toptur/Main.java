package toptur;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;
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
 * args[0] -- what task we are going to do args[1+] -- whatever input is needed
 */
public class Main {
	
	//////////////////////
	// MEMBER VARIABLES //
	//////////////////////

	public static final String TOPTUR_DATA_FOLDER = ".toptur_data/";

    public static final String DEV_DOCS = "dataset/dev";
    public static final String TEST_DOCS = "dataset/test";
    public static final String ORIGINAL_DOCS = "dataset/original_dataset/docs";

    public static final String SENTENCES_TRAINING_FILE = TOPTUR_DATA_FOLDER + "sentences_train.vector";
    public static final String SENTENCES_MODEL_FILE = TOPTUR_DATA_FOLDER + "liblinear_models/sentences.model";
    public static final String SENTENCES_TEST_FILE = TOPTUR_DATA_FOLDER + "sentences_test.vector";

	public static final String AGENT_TRAINING_FILE = TOPTUR_DATA_FOLDER + "agent_train.vector";
	public static final String AGENT_MODEL_FILE = TOPTUR_DATA_FOLDER + "liblinear_models/agent.model";
	public static final String AGENT_TEST_FILE = TOPTUR_DATA_FOLDER + "agent_test.vector";

	public static final String TARGET_TRAINING_FILE = TOPTUR_DATA_FOLDER + "target_train.vector";
	public static final String TARGET_MODEL_FILE = TOPTUR_DATA_FOLDER + ".liblinear_models/target.model";
	public static final String TARGET_TEST_FILE = TOPTUR_DATA_FOLDER + "target_test.vector";

	public static final String OPINION_TRAINING_FILE = TOPTUR_DATA_FOLDER + "opinion_train.vector";
	public static final String OPINION_TEST_FILE = TOPTUR_DATA_FOLDER + "opinion_test.vector";
	public static final String OPINION_MODEL_FILE = TOPTUR_DATA_FOLDER + "liblinear_models/opinion.model";

	public static final String LIB_LINEAR_FEATURE_MANAGER_FILE = TOPTUR_DATA_FOLDER + "/lib_linear_feature_manager.json";

    public static final String SENTI_WORD_NET_FILE = "sentiwordnet.txt";

    private static SentiWordNetDictionary sentiWordNetDictionary;
    private static volatile StanfordCoreNLP stanfordCoreNLP;

    private static final String PHI_WORD = "__PHI__";
    private static final String PHI_POS = "__PHI_POS__";
	private static final String OMEGA_WORD = "__OMEGA__";
	private static final String OMEGA_POS = "__OMEGA_POS__";


    //////////////////////
    //       MAIN       //
    //////////////////////
    
    public static void main(String[] args) {
        if (args.length == 0)
            System.exit(0);

        System.out.println("Gathering SentiWordNet dictionary...");
        getSentiWordNet();

        String task = args[0].toLowerCase();
        if (task.equals("train")) {
            ArrayList<NewsArticle> devArticles = getAllDocsFrom(DEV_DOCS);

            // Train to detect opinionated sentences
            createSentencesVectorFile(devArticles, SENTENCES_TRAINING_FILE);
            trainLibLinear(SENTENCES_TRAINING_FILE, SENTENCES_MODEL_FILE);
            testLibLinear(SENTENCES_TRAINING_FILE, SENTENCES_MODEL_FILE, "/dev/null");

			// Train to detect opinion agents
			createVectorFileAgent(devArticles, AGENT_TRAINING_FILE);
			trainLibLinear(AGENT_TRAINING_FILE, AGENT_MODEL_FILE);
            testLibLinear(AGENT_TRAINING_FILE, AGENT_MODEL_FILE, "/dev/null");

			// Train to detect opinion agents
			createVectorFileTarget(devArticles, TARGET_TRAINING_FILE);
			trainLibLinear(TARGET_TRAINING_FILE, TARGET_MODEL_FILE);
            testLibLinear(TARGET_TRAINING_FILE, TARGET_MODEL_FILE, "/dev/null");

            // Train to detect opinions in the sentence
            createOpinionVectorFile(devArticles, OPINION_TRAINING_FILE);
            trainLibLinear(OPINION_TRAINING_FILE, OPINION_MODEL_FILE);
            testLibLinear(OPINION_TRAINING_FILE, OPINION_MODEL_FILE, "/dev/null");

            LibLinearFeatureManager.saveInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

        } else if (task.equals("test")) {
        	TreeSet<EvaluationOption> evalOptions = new TreeSet<EvaluationOption>();

        	if (args.length == 1) {
        		for (EvaluationOption e : EvaluationOption.values())
        			evalOptions.add(e);
			} else {
				for (int i = 1; i < args.length; i++) {
					EvaluationOption e = EvaluationOption.valueOf(args[i].replaceAll("-", "").toUpperCase());
					evalOptions.add(e);
				}
			}

            ArrayList<NewsArticle> testArticles = getAllDocsFrom(TEST_DOCS);

            createSentencesVectorFile(testArticles, SENTENCES_TEST_FILE);
            createVectorFileAgent(testArticles, AGENT_TEST_FILE);
            createVectorFileTarget(testArticles, TARGET_TEST_FILE);
            createOpinionVectorFile(testArticles, OPINION_TEST_FILE);

            testLibLinear(SENTENCES_TEST_FILE, SENTENCES_MODEL_FILE, "/dev/null");
            testLibLinear(AGENT_TEST_FILE, AGENT_MODEL_FILE, "/dev/null");
            testLibLinear(TARGET_TEST_FILE, TARGET_MODEL_FILE, "/dev/null");
            testLibLinear(OPINION_TEST_FILE, OPINION_MODEL_FILE, "/dev/null");

            // Extract the opinions
            // Let's time how long it takes
            System.out.println("Starting extraction...");
            long startTime = System.currentTimeMillis();
            for (NewsArticle a : testArticles)
                extractAllOpinionsFor(a);
            long endTime = System.currentTimeMillis();

            System.out.println(((double) endTime - startTime) / 1000.0  + " seconds");

            evaluateExtractedOpinions(testArticles, evalOptions);

        } else if (task.equals("extract")) {
            for (int i = 1; i < args.length; i++) {
                File file = new File(args[i]);

                if (file.exists()) {
                    NewsArticle article = new NewsArticle(file);
                    extractAllOpinionsFor(article);
                    System.out.println(article);
                } else {
                    System.err.println("\"" + args[i] + "\" could not be found!");
                }
            }

        } else {
            System.err.println("No valid task provided");
            System.exit(1);
        }
    }

	/////////////////////
	// DATA PROCESSING //
	/////////////////////
    
	/**
     * Reads all docs in some path and converts them to NewsArticles.
     * 
     * Reads gold standard data and populates in NewsArticles.
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
    
    ////////////////////////
    // FEATURE PROCESSING //
    ////////////////////////
    
	/**
	 * Generates a vector file in LibLinear format for whatever articles are
	 * provided.
	 * 
	 * This vector file contains features for identifying an opinionated sentence. 
	 *
	 * @param articles
	 * @param nameOfVectorFile
	 */
	private static void createSentencesVectorFile(ArrayList<NewsArticle> articles, String nameOfVectorFile) {
		StringBuilder vectorFileBuilder = new StringBuilder();
		LibLinearFeatureManager libLinearFeatureManager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

		for (NewsArticle article : articles)
		{
			Document doc = new Document(article.getFullText());

			for (Sentence s : doc.sentences())
			{
				StringBuilder vectorLineBuilder = new StringBuilder();
				TreeMap<Integer, Object> libLinearFeatureVector = new TreeMap<Integer, Object>();

				// The label for this sentence
				if (article.sentenceHasOpinion(s.toString()))
					vectorLineBuilder.append(1);
				else
					vectorLineBuilder.append(0);

				List<String> words = s.words();

				// Creating the feature vectors
				for (LibLinearFeatureManager.LibLinearFeature feature : LibLinearFeatureManager.LibLinearFeature
						.values())
				{
					int id;

					switch (feature)
					{
					case CONTAINS_UNIGRAM:
						for (String w : words)
						{
							id = libLinearFeatureManager.getIdFor(feature, w);
							libLinearFeatureVector.put(id, true);
						}
						break;
					case CONTAINS_BIGRAM:
						String bigram;

						for (int i = 0; i < words.size() + 1; i++) {
							if (i == 0) {
								bigram = PHI_WORD + " " + words.get(i);
							} else if (i == words.size()) {
								bigram = words.get(i - 1) + " " + OMEGA_WORD;
							} else {
								bigram = words.get(i - 1) + " " + words.get(i);
							}

							id = libLinearFeatureManager.getIdFor(feature, bigram);
							libLinearFeatureVector.put(id, true);
						}
						break;
					case OBJECTIVITY_OF_SENTENCE:
						id = libLinearFeatureManager.getIdFor(feature, "");
						double objectivity = 0.0;

						for (String w : words)
						{
							objectivity += sentiWordNetDictionary.getObjectivityOf(w);
						}

						objectivity /= words.size();
						libLinearFeatureVector.put(id, objectivity);

						break;

					// case HAS_WORD_RELATED_TO_OTHER_WORD:
					// for (String w : words) {
					// DataMuseWord[] dataMuseWords = DataMuse.getWordsRelatedTo(w);
					// if (dataMuseWords == null)
					// continue;
					//
					// for (DataMuseWord dmw : dataMuseWords) {
					// id = libLinearFeatureManager.getIdFor(feature, dmw.word);
					// libLinearFeatureVector.put(id, true);
					// }
					// }
					// break;
					}
				}

				for (Map.Entry<Integer, Object> e : libLinearFeatureVector.entrySet())
				{
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

		try
		{
			PrintWriter vectorFile = new PrintWriter(nameOfVectorFile);
			vectorFile.print(vectorFileBuilder.toString());
			vectorFile.flush();
			vectorFile.close();
		} catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}
	}

	/**
     * Generates a vector file in LibLinear format for whatever articles are provided.
     *
     * This vector file contains features for identifying the Agent of an opinion. 
     *
     * @param articles
     * @param nameOfVectorFile
     */
    private static void createVectorFileAgent(ArrayList<NewsArticle> articles, String nameOfVectorFile) {
        StringBuilder vectorFileBuilder = new StringBuilder();
        LibLinearFeatureManager libLinearFeatureManager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

        for (NewsArticle article : articles) {
            Document doc = new Document(article.getFullText());

            for (Sentence s : doc.sentences()) {
				if (article.sentenceHasOpinion(s.toString())) {

					String agent = article.getOpinionAgent(s.toString());

					for(String word : s.words()) {

						StringBuilder vectorLineBuilder = new StringBuilder();
						TreeMap<Integer, Object> libLinearFeatureVector = new TreeMap<Integer, Object>();

						// The label for this sentence
						if (word.equalsIgnoreCase(agent))
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


    /**
     * Generates a vector file in LibLinear format for whatever articles are provided.
     * 
     * This vector file contains features for identifying the Target of an opinion. 
     * 
     * @param articles
     * @param nameOfVectorFile
     */
    private static void createVectorFileTarget(ArrayList<NewsArticle> articles, String nameOfVectorFile) {
        StringBuilder vectorFileBuilder = new StringBuilder();
        LibLinearFeatureManager libLinearFeatureManager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

        for (NewsArticle article : articles) {
            Document doc = new Document(article.getFullText());

            for (Sentence s : doc.sentences()) {

            		if (article.sentenceHasOpinion(s.toString())) {

            			String agent = article.getOpinionAgent(s.toString());

            			for(String word : s.words()) {

		                StringBuilder vectorLineBuilder = new StringBuilder();
		                TreeMap<Integer, Object> libLinearFeatureVector = new TreeMap<Integer, Object>();

		                // The label for this sentence
		                if (word.equalsIgnoreCase(agent))
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

    /**
     * Generates a vector file in LibLinear format for whatever articles are provided.
     * 
     * This vector file contains features for identifying the Target of an opinion. 
     * 
     * @param sentence, nameOfVectorFile
     * @param nameOfVectorFile
     */
	private static void createSingleSentenceVectorFile(Sentence sentence, String nameOfVectorFile) {
		StringBuilder vectorFileBuilder = new StringBuilder();
		LibLinearFeatureManager libLinearFeatureManager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

		StringBuilder vectorLineBuilder = new StringBuilder();
		TreeMap<Integer, Object> libLinearFeatureVector = new TreeMap<Integer, Object>();

		vectorFileBuilder.append("0"); // Assume objective

		List<String> words = sentence.words();

		// Creating the feature vectors
		for (LibLinearFeatureManager.LibLinearFeature feature : LibLinearFeatureManager.LibLinearFeature.values())
		{
			switch (feature)
			{
			case CONTAINS_UNIGRAM:
				for (String w : words)
				{
					int id = libLinearFeatureManager.getIdFor(feature, w);
					libLinearFeatureVector.put(id, true);
				}
				break;
			case CONTAINS_BIGRAM:
				String bigram;

				for (int i = 0; i < words.size() + 1; i++) {
					if (i == 0) {
						bigram = PHI_WORD + " " + words.get(i);
					} else if (i == words.size()) {
						bigram = words.get(i - 1) + " " + OMEGA_WORD;
					} else {
						bigram = words.get(i - 1) + " " + words.get(i);
					}

					int id = libLinearFeatureManager.getIdFor(feature, bigram);
					libLinearFeatureVector.put(id, true);
				}
				break;
			case OBJECTIVITY_OF_SENTENCE:
				int id = libLinearFeatureManager.getIdFor(feature, "");
				double objectivity = 0.0;

				for (String w : words)
				{
					objectivity += sentiWordNetDictionary.getObjectivityOf(w);
				}

				objectivity /= words.size();
				libLinearFeatureVector.put(id, objectivity);

				break;
			}
		}

		for (Map.Entry<Integer, Object> e : libLinearFeatureVector.entrySet())
		{
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

		try
		{
			PrintWriter vectorFile = new PrintWriter(nameOfVectorFile);
			vectorFile.print(vectorFileBuilder.toString());
			vectorFile.flush();
			vectorFile.close();
		} catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}
	}


	/**
	 * Creates a vector file for a single word. Use for BIO labeling.
	 *
	 * @param previousWord
	 * @param thisWord
	 * @param nextWord
	 * @param previousPos
	 * @param thisPos
	 * @param nextPos
	 * @param fileName
	 */
	private static void createSingleWordVectorFile(String previousWord, String thisWord, String nextWord, String previousPos, String thisPos, String nextPos, String fileName) {
	    LibLinearFeatureManager manager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);
	    StringBuilder vectorFileBuilder = new StringBuilder();
	    vectorFileBuilder.append(0);
	    vectorFileBuilder.append(' ');

        TreeMap<Integer, String> stupidMap = new TreeMap<Integer, String>();
        for (LibLinearFeatureManager.LibLinearFeature feature : LibLinearFeatureManager.LibLinearFeature.values()) {
            int id;
            switch(feature) {
				case PREVIOUS_UNIGRAM:
					id = manager.getIdFor(feature, previousWord);
					stupidMap.put(id, id + ":1");
					break;
                case THIS_UNIGRAM:
                    id = manager.getIdFor(feature, thisWord);
                    stupidMap.put(id, id + ":1");
                    break;
				case NEXT_UNIGRAM:
					id = manager.getIdFor(feature, nextWord);
					stupidMap.put(id, id + ":1");
					break;
				case PREVIOUS_PART_OF_SPEECH:
					id = manager.getIdFor(feature, previousPos);
					stupidMap.put(id, id + ":1");
					break;
                case THIS_PART_OF_SPEECH:
                    id = manager.getIdFor(feature, thisPos);
                    stupidMap.put(id, id + ":1");
                    break;
				case NEXT_PART_OF_SPEECH:
					id = manager.getIdFor(feature, nextPos);
					stupidMap.put(id, id + ":1");
					break;
                case OBJECTIVITY_OF_WORD:
                    id = manager.getIdFor(feature, "");
                    double objectivity = sentiWordNetDictionary.getObjectivityOf(thisWord);
                    stupidMap.put(id, id + ":" + objectivity);
                    break;
            }
        }

        for (String s : stupidMap.values()) {
            vectorFileBuilder.append(s);
            vectorFileBuilder.append(' ');
        }

		try
		{
			PrintWriter vectorFile = new PrintWriter(fileName);
			vectorFile.print(vectorFileBuilder.toString());
			vectorFile.flush();
			vectorFile.close();
		} catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}
    }

	/**
	 * Creates the opinion vector file that uses sequence labeling for opinions.
	 *
	 * @param articles
	 * @param opinionTrainingFile
	 */
	private static void createOpinionVectorFile(ArrayList<NewsArticle> articles, String opinionTrainingFile) {
	    LibLinearFeatureManager manager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);
        StringBuilder vectorFileBuilder = new StringBuilder();

	    // Here we're going to do your standard BIO labeling to extract the opinion
	    for (NewsArticle a : articles) {

			// Get a mapping of sentences to all of their opinions
			HashMap<String, HashSet<Opinion>> opinionatedSentences = new HashMap<String, HashSet<Opinion>>();

			for (Opinion o : a.getGoldStandardOpinions().values()) {
				if (opinionatedSentences.containsKey(o.sentence)) {
					opinionatedSentences.get(o.sentence).add(o);
				} else {
					HashSet<Opinion> set = new HashSet<Opinion>();
					set.add(o);
					opinionatedSentences.put(o.sentence, set);
				}
			}

			// Now let's figure out where the B and I labels go. Every other word will have an O label
			for (Map.Entry<String, HashSet<Opinion>> e : opinionatedSentences.entrySet()) {
				Sentence sentence = new Document(e.getKey()).sentence(0);

				TreeSet<String> opinionExpressions = new TreeSet<String>();
				for (Opinion o : e.getValue())
					opinionExpressions.add(o.opinion);

				TreeMap<Integer, Integer> sentenceIndexToBIOLabel = new TreeMap<Integer, Integer>();

				// Assign BIO labels
				for (String opinionExpression : opinionExpressions) {
					String[] expression = opinionExpression.split(" ");
					String firstExpressionWord = expression[0];

					for (int i = 0; i < sentence.words().size(); i++) {
						String word = sentence.word(i);

						if (firstExpressionWord.equals(word)) {
							boolean found = true;

							int j = 1;
							for (int k = i + j; k < sentence.words().size() && j < expression.length; k++) {
								if (!expression[j].equals(sentence.word(k))) {
									found = false;
									break;
								}

								j += 1;
							}

							if (found) {
								sentenceIndexToBIOLabel.put(i, 1);

								for (j = 1; j < expression.length; j++)
									sentenceIndexToBIOLabel.put(i + j, 2);
							}
						}
					}
				}

				// And now, setup the vectors
				for (int i = 0; i < sentence.words().size(); i++) {
					String word = sentence.word(i);
					String pos = sentence.posTag(i);

					TreeMap<Integer, String> stupidMap = new TreeMap<Integer, String>();
					for (LibLinearFeatureManager.LibLinearFeature feature : LibLinearFeatureManager.LibLinearFeature.values()) {
						int id;
						switch (feature) {
							case PREVIOUS_UNIGRAM:
								String previous;

								if (i > 0)
									previous = sentence.word(i - 1);
								else
									previous = PHI_WORD;

								id = manager.getIdFor(feature, previous);
								stupidMap.put(id, id + ":1");
								break;
							case THIS_UNIGRAM:
								id = manager.getIdFor(feature, word);
								stupidMap.put(id, id + ":1");
								break;
							case NEXT_UNIGRAM:
								String next;

								if (i < sentence.words().size() - 1)
									next = sentence.word(i + 1);
								else
									next = OMEGA_WORD;

								id = manager.getIdFor(feature, next);
								stupidMap.put(id, id + ":1");
								break;
							case PREVIOUS_PART_OF_SPEECH:
								String previousPos;

								if (i > 0)
									previousPos = sentence.posTag(i - 1);
								else
									previousPos = PHI_POS;

								id = manager.getIdFor(feature, previousPos);
								stupidMap.put(id, id + ":1");
								break;
							case THIS_PART_OF_SPEECH:
								id = manager.getIdFor(feature, pos);
								stupidMap.put(id, id + ":1");
								break;
							case NEXT_PART_OF_SPEECH:
								String nextPos;

								if (i < sentence.words().size() - 1)
									nextPos = sentence.posTag(i + 1);
								else
									nextPos = OMEGA_POS;

								id = manager.getIdFor(feature, nextPos);
								stupidMap.put(id, id + ":1");
								break;
							case OBJECTIVITY_OF_WORD:
								id = manager.getIdFor(feature, "");
								double objectivity = sentiWordNetDictionary.getObjectivityOf(word);
								stupidMap.put(id, id + ":" + objectivity);
								break;
						}
					}

					if (sentenceIndexToBIOLabel.containsKey(i)) {
						vectorFileBuilder.append(sentenceIndexToBIOLabel.get(i));
					} else {
						vectorFileBuilder.append(0);
					}

					vectorFileBuilder.append(' ');

					for (String s : stupidMap.values()) {
						vectorFileBuilder.append(s);
						vectorFileBuilder.append(' ');
					}

					vectorFileBuilder.append('\n');
				}
			}
		}

		// TODO: Should we do this with non-opinionated sentences as well?

        try {
            PrintWriter vectorFile = new PrintWriter(opinionTrainingFile);
            vectorFile.print(vectorFileBuilder.toString().trim());
            vectorFile.flush();
            vectorFile.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

	///////////////////////
	// TRAIN CLASSIFIERS //
	///////////////////////
	
	private static void trainLibLinear(String vectorFileName, String modelFileName) {
		// Now that the vector file is put together, we need to run liblinear
		try
		{
			Runtime.getRuntime().exec("./liblinear_train " + vectorFileName + " " + modelFileName);
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	//////////////////////
	// TEST CLASSIFIERS //
	//////////////////////
	
	private static void testLibLinear(String testVectorFileName, String modelFileName, String outputFileName) {
		// Now that the vector file is put together, we need to run liblinear
		try
		{
			System.out.println("Predicting \"" + testVectorFileName + "\" with liblinear...");

			Runtime runtime = Runtime.getRuntime();
			Scanner s = new Scanner(runtime
					.exec("./liblinear_predict " + testVectorFileName + " " + modelFileName + " " + outputFileName)
					.getInputStream());
			Thread.sleep(20);
			while (s.hasNextLine())
				System.out.println(s.nextLine());

		} catch (IOException e)
		{
			e.printStackTrace();
		} catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

	//////////////////////////////////////
	// EXTRACT/EVALUATE WITH CLASSIFIER //
	//////////////////////////////////////
	
	// Evaluate
	private static boolean sentenceContainsOpinion(Sentence sentence) {
		String name = "some_file.vector";
		createSingleSentenceVectorFile(sentence, name);

		try
		{
			Runtime.getRuntime().exec("./liblinear_predict " + name + " " + SENTENCES_MODEL_FILE + " output.txt");
			Thread.sleep(100); // To give LibLinear enough time to output to file
			Scanner derp = new Scanner(Runtime.getRuntime().exec("cat output.txt").getInputStream());
			int i = derp.nextInt();
			derp.close();
			return i == 1;
		} catch (IOException e)
		{
			e.printStackTrace();
		} catch (InterruptedException e)
		{
			e.printStackTrace();
		}

		return false;
	}
		
	// This is for evaluating 
    private static void extractAllOpinionsFor(NewsArticle article) {
        Document doc = new Document(article.getFullText());

        for (Sentence s : doc.sentences()) {

            if (sentenceContainsOpinion(s)) {
            	for (String opinionExpression : extractAllOpinionsFrom(s)) {
            		Opinion o = new Opinion();
            		o.opinion = opinionExpression;
            		o.sentence = s.toString();

            		// TODO: Add agent/target/polarity

					article.addExtractedOpinion(o);
				}
            }
        }
    }

    /**
     * Extracts all opinions from the sentence provided.
     *
     * @param s - Sentence to get opinions from
     * @return Set of type String
     */
	private static Set<String> extractAllOpinionsFrom(Sentence s) {
        String name = "some_file.vector";

        int previousTag = 0;

        TreeSet<String> allOpinions = new TreeSet<String>();

        StringBuilder opinionBuilder = new StringBuilder();
        for (int i = 0; i < s.words().size(); i++) {
        	String previousWord, nextWord, previousPos, nextPos;

        	if (i > 0) {
        		previousWord = s.word(i - 1);
        		previousPos = s.posTag(i - 1);
			} else {
        		previousWord = PHI_WORD;
				previousPos = PHI_POS;
			}

			if (i < s.words().size() - 1) {
        		nextWord = s.word(i + 1);
        		nextPos = s.posTag(i + 1);
			} else {
        		nextWord = OMEGA_WORD;
        		nextPos = OMEGA_POS;
			}

            String word = s.word(i);
            String pos = s.posTag(i);
            createSingleWordVectorFile(previousWord, word, nextWord, previousPos, pos, nextPos, name);

            try {
                Runtime.getRuntime().exec("./liblinear_predict " + name + " " + SENTENCES_MODEL_FILE + " output.txt");
                Thread.sleep(100); // To give LibLinear enough time to output to file

				Scanner scanner = new Scanner(new File("output.txt"));

				while(scanner.hasNextLine()) {
					int label = Integer.parseInt(scanner.nextLine());

					if (label != 0) {
						opinionBuilder.append(word);
						opinionBuilder.append(" ");
					} else if (opinionBuilder.length() > 0) {
						allOpinions.add(opinionBuilder.toString());
						opinionBuilder = new StringBuilder();
					}
				}
            } catch (IOException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        return allOpinions;
	}
	
	// Evaluate
	private static void evaluateExtractedOpinions(ArrayList<NewsArticle> articles, Set<EvaluationOption> evaluationOptions) {
		double totalFScore = 0.0;

		System.out.println("Name\tPrecision\tRecall\tFScore");

		for (NewsArticle article : articles) {
			HashMap<String, Opinion> extractedOpinions = (HashMap<String, Opinion>) article.getExtractedOpinions().clone();
			HashMap<String, Opinion> goldStandardOpinions = (HashMap<String, Opinion>) article.getGoldStandardOpinions().clone();

			double truePositives = 0.0;

			TreeSet<String> correct = new TreeSet<String>();

			for (Opinion extracted : extractedOpinions.values()) {
				for (Opinion goldStandard : goldStandardOpinions.values()) {
					boolean matches = true;

					for (EvaluationOption option : evaluationOptions) {
						if (!Opinion.opinionsMatchGivenOption(extracted, goldStandard, option)) {
							matches = false;
							break;
						}
					}

					if (matches && !correct.contains(extracted.opinion)) {
						correct.add(extracted.opinion);
						truePositives += 1.0;
					}
				}
            }

            for (String s : correct) {
				extractedOpinions.remove(s);
				goldStandardOpinions.remove(s);
            }

			double precision = truePositives / (extractedOpinions.size() + correct.size());
			double recall = truePositives / (goldStandardOpinions.size() + correct.size());
			double fscore = 2 * ((precision * recall) / (Math.max(1, precision + recall)));

			System.out.println(article.getDocumentName() + "\t" + precision + "\t" + recall + "\t" + fscore);
			totalFScore += fscore;
		}

		System.out.println("\nTOTAL FSCORE: " + totalFScore / articles.size());
	}
	
	////////////////////
	// HELPER METHODS //
	////////////////////
	
	private static void getSentiWordNet() {
		sentiWordNetDictionary = new SentiWordNetDictionary();

		try
		{
			Scanner s = new Scanner(new File(SENTI_WORD_NET_FILE));
			while (s.hasNextLine())
			{
				String line = s.nextLine();

				if (line.startsWith("#"))
					continue;

				String[] lineParts = line.split("\t");

				if (lineParts.length != 6)
				{
					System.out.println(line);
					throw new RuntimeException(
							"One of the lines in the SENTI_WORD_NET_FILE was not formatted properly!");
				}

				double positive = Double.parseDouble(lineParts[2]);
				double negative = Double.parseDouble(lineParts[3]);

				String[] words = lineParts[4].split(" ");
				for (String w : words)
				{
					sentiWordNetDictionary.addWord(w.split("#")[0].replaceAll("_", " "), positive, negative);
				}
			}
		} catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}
	}
	
	
}
