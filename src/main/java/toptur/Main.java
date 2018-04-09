package toptur;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
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
 * args[0] -- what task we are going to do
 * args[1+] -- whatever input is needed
 *
 * So like before, we will have a few different classifiers that each define various labels to denote whether something
 * belongs to a certain label. So to see if a sentence is opinionated or not, we will have one learner to figure that
 * out on a sentence level, another to get all the opinions in the sentence, etc.
 */
public class Main
{

	//////////////////////
	//    VARIABLES     //
	//////////////////////

	public static final String TOPTUR_DATA_FOLDER = ".toptur_data/";

	public static final String DEV_DOCS = "dataset/dev";
	public static final String TEST_DOCS = "dataset/test";
	public static final String ORIGINAL_DOCS = "dataset/original_dataset/docs";

	public static final String SENTI_WORD_NET_FILE = "sentiwordnet.txt";
	public static final String RELATED_WORDS_FILE = TOPTUR_DATA_FOLDER + "related_words.json";
	public static final String LIB_LINEAR_FEATURE_MANAGER_FILE = TOPTUR_DATA_FOLDER + "lib_linear_feature_manager.json";
	public static final String LEARNER_FEATURE_MANAGER_FILE = TOPTUR_DATA_FOLDER + "learner_feature_manager.json";

	public static final String BAGGED_TREES_SENTENCE_CLASSIFIER = TOPTUR_DATA_FOLDER + "bagged_trees_sentences.json";
	public static final String BAGGED_TREES_OPINION_CLASSIFIER = TOPTUR_DATA_FOLDER + "bagged_trees_opinions.json";
	public static final String BAGGED_TREES_AGENT_CLASSIFIER = TOPTUR_DATA_FOLDER + "bagged_trees_agents.json";
	public static final String BAGGED_TREES_TARGET_CLASSIFIER = TOPTUR_DATA_FOLDER + "bagged_trees_targets.json";
	public static final String BAGGED_TREES_POLARITY_CLASSIFIER = TOPTUR_DATA_FOLDER + "bagged_trees_polarity.json";

	public static final String SENTENCES_LIB_LINEAR_MODEL_FILE = TOPTUR_DATA_FOLDER + "lib_linear_sentences.model";
	public static final String OPINIONS_LIB_LINEAR_MODEL_FILE = TOPTUR_DATA_FOLDER + "lib_linear_opinons.model";
	public static final String TARGET_LIB_LINEAR_MODEL_FILE = TOPTUR_DATA_FOLDER + "lib_linear_targets.model";
	public static final String AGENT_LIB_LINEAR_MODEL_FILE = TOPTUR_DATA_FOLDER + "lib_linear_agents.model";
	public static final String POLARITY_LIB_LINEAR_MODEL_FILE = TOPTUR_DATA_FOLDER + "lib_linear_polarity.model";

	private static final String PHI_WORD = "_PHI_WORD_";
	private static final String OMEGA_WORD = "_OMEGA_WORD_";
	private static final String PHI_POS = "_PHI_POS_";
	private static final String OMEGA_POS = "_OMEGA_POS_";

	private static SentiWordNetDictionary sentiWordNetDictionary;
	private static Map<String, String[]> relatedWordsMap;
	private static Set<String> allWords;
	private static Set<String> allPos;

	// TODO: Set these to something that gives at least decent performance. I'm keeping them low to fix any potential bugs we have
	private static int NUMBER_OF_TREES = 10;
	private static int DEPTH_OF_TREES = 2;

	//////////////////////
	//       MAIN       //
	//////////////////////

	public static void main(String[] args)
	{
		if (args.length == 0)
			System.exit(0);

		System.out.println("Gathering SentiWordNet dictionary...");
		getSentiWordNet();
		getRelatedWordsMap();
		getLearnerFeatureManager();

		String task = args[0].toLowerCase();

		if (task.equals("train"))
		{
			ArrayList<NewsArticle> devArticles = getAllDocsFrom(DEV_DOCS);

			// TODO: Multithread this chunk

			// Train the sentence classifier
//			System.out.println("Training the sentence classifier!");
//			System.out.print("\tbagged trees...");
//			List<LearnerExample<Sentence, Boolean>> sentenceExamples = getOpinionatedSentenceExamples(devArticles);
//			BaggedTrees<Sentence, Boolean> opinionatedSentenceClassifier = new BaggedTrees<Sentence, Boolean>(sentenceExamples,
//					LearnerFeatureManager.getInstance(LEARNER_FEATURE_MANAGER_FILE).getIdsFor(LearnerFeature.getSentenceFeatures()), NUMBER_OF_TREES, DEPTH_OF_TREES);
//			System.out.println("done.");
//			System.out.print("\tliblinear...");
//			createSentencesVectorFile(devArticles, ".sentences.vector", opinionatedSentenceClassifier);
//			trainLibLinear(".sentences.vector", SENTENCES_LIB_LINEAR_MODEL_FILE);
//			System.out.println("done.");

			System.out.println("Training the opinion classifier!");
			System.out.print("\tbagged trees...");
			List<LearnerExample<String, Integer>> opinionExamples = getOpinionWordExamples(devArticles);
			BaggedTrees<String, Integer> opinionatedWordClassifier = new BaggedTrees<String, Integer>(opinionExamples,
					LearnerFeatureManager.getInstance(LEARNER_FEATURE_MANAGER_FILE).getIdsFor(LearnerFeature.getOpinionPhraseFeatures()), NUMBER_OF_TREES, DEPTH_OF_TREES);
			System.out.println("done.");
			System.out.print("\tliblinear...");
			createOpinionatedPhraseVectorFile(devArticles, ".opinions.vector", opinionatedWordClassifier);
			trainLibLinear(".opinions.vector", OPINIONS_LIB_LINEAR_MODEL_FILE);
			System.out.println("done.");

			// TODO: Train the agent classifier
			// TODO: Train the target classifier
			// TODO: Train the polarity classifier

			// TODO: Test accuracy of all other learners
			// TODO: Train liblinear to handle input from the bagged tree classifier
			// TODO: Train all the other learners

			System.out.println("Saving classifiers to disk...");
//			opinionatedSentenceClassifier.saveToFile(BAGGED_TREES_SENTENCE_CLASSIFIER);
			opinionatedWordClassifier.saveToFile(BAGGED_TREES_OPINION_CLASSIFIER);
			LearnerFeatureManager.getInstance(null).saveInstance(LEARNER_FEATURE_MANAGER_FILE);
			LibLinearFeatureManager.saveInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

            System.out.println("Finished Training");

		} else if (task.equals("test"))
		{
			TreeSet<EvaluationOption> evalOptions = new TreeSet<EvaluationOption>();

			if (args.length == 1)
			{
				evalOptions.addAll(Arrays.asList(EvaluationOption.values()));
			} else
			{
				for (int i = 1; i < args.length; i++)
				{
					EvaluationOption e = EvaluationOption.valueOf(args[i].replaceAll("-", "").toUpperCase());
					evalOptions.add(e);
				}
			}

			ArrayList<NewsArticle> testArticles = getAllDocsFrom(TEST_DOCS);

			// TODO: Test learners

		} else if (task.equals("extract"))
		{
			// TODO: Extract things

		} else if (task.equals("explore"))
		{
			ArrayList<NewsArticle> devArticles = getAllDocsFrom(DEV_DOCS);

			for (NewsArticle article : devArticles)
			{
				HashMap<String, Opinion> gold = article.getGoldStandardOpinions();
				for (String key : gold.keySet())
				{
					if (gold.get(key).agent != null && !gold.get(key).agent.equals("w"))
					{
						System.out.println("Sentence: " + gold.get(key).sentence);
						System.out.println("Opinion: " + gold.get(key).opinion);
						System.out.println("Agent: " + gold.get(key).agent);
						System.out.println("Target: " + gold.get(key).target);
						System.out.println("Sentiment: " + gold.get(key).sentiment);
						System.out.println();
					}
				}
			}
		} else
		{
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
	 * @param path -
	 * @return __
	 */
	private static ArrayList<NewsArticle> getAllDocsFrom(String path)
	{
		ArrayList<NewsArticle> docs = new ArrayList<NewsArticle>();

		File folder = new File(path);

		if (!folder.exists())
		{
			System.err.println("Could not find folder \"" + path + "\"!");
			System.exit(1);
		}

		for (File f : folder.listFiles())
		{
			docs.add(NewsArticle.fromJson(f));
		}

		if (allWords == null)
			allWords = new TreeSet<String>();
		if (allPos == null)
			allPos = new TreeSet<String>();

		for (NewsArticle a : docs) {
			for (Sentence s : new Document(a.getFullText()).sentences()) {
				allWords.addAll(s.words());
				allPos.addAll(s.posTags());
			}
		}

		return docs;
	}

	/**
	 * Helper method. Takes a collection of articles and returns a collection of learner examples to train a classifier
	 * that detects opinionated sentences.
	 *
	 * @param articles - articles to get examples from
	 * @return
	 */
	private static List<LearnerExample<Sentence, Boolean>> getOpinionatedSentenceExamples(ArrayList<NewsArticle> articles) {
		List<LearnerExample<Sentence, Boolean>> opinionatedSentenceExamples = new ArrayList<LearnerExample<Sentence, Boolean>>();

		LearnerExampleValueRequestListener<Sentence> listener = new LearnerExampleValueRequestListener<Sentence>() {
			public Object valueOfFeatureForExample(Sentence example, int featureId) {
				LearnerFeature f = LearnerFeatureManager.getInstance(LEARNER_FEATURE_MANAGER_FILE).getLearnerFeatureFor(featureId);
				Object val = LearnerFeatureManager.getInstance(LEARNER_FEATURE_MANAGER_FILE).getValueFor(featureId);

				// TODO: Add all the other features
				switch(f) {
					case CONTAINS_UNIGRAM:
						String unigram = (String) val;
						return example.words().contains(unigram);
					case CONTAINS_BIGRAM:
						String bigram = (String) val;

						for (int i = 0; i <= example.words().size(); i++) {
							String tmp;

							if (i == 0)
								tmp = PHI_WORD + " " + example.word(i);
							else if (i == example.words().size())
								tmp = example.word(i - 1) + " " + OMEGA_WORD;
							else
								tmp = example.word(i - 1) + " " + example.word(i);

							if (tmp.equalsIgnoreCase(bigram))
								return true;
						}

						return false;
					default:
						;
				}

				throw new RuntimeException("Could not assign a label for the feature provided!");
			}
		};

		for (NewsArticle a : articles) {
			Document document = new Document(a.getFullText());

			for (Sentence s : document.sentences()) {
				LearnerExample<Sentence, Boolean> example = new LearnerExample<Sentence, Boolean>(s, a.sentenceHasOpinion(s.toString()), listener);
				opinionatedSentenceExamples.add(example);
			}
		}

		return opinionatedSentenceExamples;
	}


	/**
	 * Get a collection of LearnerExamples to do BIO labeling on all of the words of an opinionated sentence.
	 *
	 * @param articles -
	 * @return List
	 */
	private static List<LearnerExample<String, Integer>> getOpinionWordExamples(ArrayList<NewsArticle> articles) {
		// TODO: This
		List<LearnerExample<String, Integer>> opinionatedPhraseExamples = new ArrayList<LearnerExample<String, Integer>>();

		LearnerExampleValueRequestListener<String> listener = new LearnerExampleValueRequestListener<String>() {
			public Object valueOfFeatureForExample(String example, int featureId) {
				LearnerFeature f = LearnerFeatureManager.getInstance(LEARNER_FEATURE_MANAGER_FILE).getLearnerFeatureFor(featureId);
				Object val = LearnerFeatureManager.getInstance(LEARNER_FEATURE_MANAGER_FILE).getValueFor(featureId);

				switch (f) {
					case THIS_WORD:
						return example;
					case THIS_POS:
						return new Document(example).sentence(0).posTag(0);
					case OBJECTIVITY_OF_WORD:
						return sentiWordNetDictionary.getObjectivityOf(example);

				}

				throw new RuntimeException("Could not get a word");
			}
		};

		// Convert dataset to something we can handle
		Map<String, Set<String>> opinionatedSentencesToOpinionPhrases = new TreeMap<String, Set<String>>();

		for (NewsArticle a : articles) {
			for (Opinion o : a.getGoldStandardOpinions().values()) {
				if (!opinionatedSentencesToOpinionPhrases.containsKey(o.sentence))
					opinionatedSentencesToOpinionPhrases.put(o.sentence, new TreeSet<String>());

				opinionatedSentencesToOpinionPhrases.get(o.sentence).add(o.opinion);
			}
		}

		for (Map.Entry<String, Set<String>> e : opinionatedSentencesToOpinionPhrases.entrySet()) {
			Sentence s = new Document(e.getKey()).sentence(0);

			// Get the right labels for each word
			for (String p : e.getValue()) {
				int firstIndex = -1;
				int lastIndex = -1;

				String[] phrase = p.split(" ");

				for (int i = 0; i < s.words().size(); i++) {
					String w = s.word(i);

					if (w.equals(phrase[0])) {
						boolean foundIt = true;

						for (int j = 1; j < phrase.length; j++) {
							if (!s.word(i + j).equals(phrase[j])) {
								foundIt = false;
								break;
							}
						}

						if (foundIt) {
							firstIndex = i;
							lastIndex = i + phrase.length - 1;
							break;
						}
					}
				}

				for (int i = 0; i < s.words().size(); i++) {
					int label = 0;

					if (i == firstIndex)
						label = 1;
					else if (i > firstIndex && i < lastIndex)
						label = 2;

					opinionatedPhraseExamples.add(new LearnerExample<String, Integer>(s.word(i), label, listener));
				}
			}
		}

		return opinionatedPhraseExamples;
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
	 * @param articles -
	 * @param nameOfVectorFile -
	 */
	private static void createSentencesVectorFile(ArrayList<NewsArticle> articles, String nameOfVectorFile, BaggedTrees<Sentence, Boolean> classifier) {
		StringBuilder vectorFileBuilder = new StringBuilder();

		for (NewsArticle article : articles)
		{
			Document doc = new Document(article.getFullText());

			for (Sentence s : doc.sentences())
			{
				StringBuilder vectorLineBuilder = new StringBuilder();

				// The label for this sentence
				if (article.sentenceHasOpinion(s.toString()))
					vectorLineBuilder.append(1);
				else
					vectorLineBuilder.append(0);

				String line = generateSentenceLineVectorFileString(article, s, classifier);
				vectorLineBuilder.append(' ');
				vectorLineBuilder.append(line);

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
	 * Creates opinionated phrase vector file
	 *
	 * @param articles -
	 * @param nameOfVectorFile -
	 * @param classifier -
	 */
	private static void createOpinionatedPhraseVectorFile(ArrayList<NewsArticle> articles, String nameOfVectorFile,
														  BaggedTrees<String, Integer> classifier) {
		LibLinearFeatureManager manager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);
		StringBuilder vectorFileBuilder = new StringBuilder();

		// Here we're going to do your standard BIO labeling to extract the opinion
		for (NewsArticle a : articles)
		{

			// Get a mapping of sentences to all of their opinions
			HashMap<String, HashSet<Opinion>> opinionatedSentences = new HashMap<String, HashSet<Opinion>>();

			for (Opinion o : a.getGoldStandardOpinions().values())
			{
				if (opinionatedSentences.containsKey(o.sentence))
				{
					opinionatedSentences.get(o.sentence).add(o);
				} else
				{
					HashSet<Opinion> set = new HashSet<Opinion>();
					set.add(o);
					opinionatedSentences.put(o.sentence, set);
				}
			}

			// Now let's figure out where the B and I labels go. Every other word will have
			// an O label
			for (Map.Entry<String, HashSet<Opinion>> e : opinionatedSentences.entrySet())
			{
				Sentence sentence = new Document(e.getKey()).sentence(0);

				TreeSet<String> opinionExpressions = new TreeSet<String>();
				for (Opinion o : e.getValue())
					opinionExpressions.add(o.opinion);

				TreeMap<Integer, Integer> sentenceIndexToBIOLabel = new TreeMap<Integer, Integer>();

				// Assign BIO labels
				for (String opinionExpression : opinionExpressions)
				{
					String[] expression = opinionExpression.split(" ");
					String firstExpressionWord = expression[0];

					for (int i = 0; i < sentence.words().size(); i++)
					{
						String word = sentence.word(i);

						if (firstExpressionWord.equals(word))
						{
							boolean found = true;

							int j = 1;
							for (int k = i + j; k < sentence.words().size() && j < expression.length; k++)
							{
								if (!expression[j].equals(sentence.word(k)))
								{
									found = false;
									break;
								}

								j += 1;
							}

							if (found)
							{
								sentenceIndexToBIOLabel.put(i, 1);

								for (j = 1; j < expression.length; j++)
									sentenceIndexToBIOLabel.put(i + j, 2);
							}
						}
					}
				}

				// And now, setup the vectors
				for (int i = 0; i < sentence.words().size(); i++)
				{
					String word = sentence.word(i);
					String pos = sentence.posTag(i);

					TreeMap<Integer, String> stupidMap = new TreeMap<Integer, String>();
					for (LibLinearFeatureManager.LibLinearFeature feature : LibLinearFeatureManager.LibLinearFeature
							.values())
					{
						int id;
						switch (feature)
						{
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
								int objectivity = sentiWordNetDictionary.getObjectivityOf(word);
								stupidMap.put(id, id + ":" + objectivity);
								break;
							case BAGGED_TREE_VOTER:
								ArrayList<NewsArticle> tmp = new ArrayList<NewsArticle>();
								tmp.add(a);
								LearnerExample<String, Integer> example = getOpinionWordExamples(tmp).get(0);
								List<Integer> guesses = classifier.allGuessesFor(example);

								for (int j = 0; j < guesses.size(); j++) {
									id = manager.getIdFor(feature, j);
									int g = guesses.get(j);
									stupidMap.put(id, id + ":" + g);
								}
								break;
						}
					}

					if (sentenceIndexToBIOLabel.containsKey(i))
					{
						vectorFileBuilder.append(sentenceIndexToBIOLabel.get(i));
					} else
					{
						vectorFileBuilder.append(0);
					}

					vectorFileBuilder.append(' ');

					for (String s : stupidMap.values())
					{
						vectorFileBuilder.append(s);
						vectorFileBuilder.append(' ');
					}

					vectorFileBuilder.append('\n');
				}
			}
		}

		try
		{
			PrintWriter vectorFile = new PrintWriter(nameOfVectorFile);
			vectorFile.print(vectorFileBuilder.toString().trim());
			vectorFile.flush();
			vectorFile.close();
		} catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}
	}

	/**
	 * Creates a single line for LibLinear of the given sentence from the news article provided.
	 *
	 * @param article -
	 * @param sentence -
	 * @return String for liblinear to mange
	 */
	private static String generateSentenceLineVectorFileString(NewsArticle article, Sentence sentence, BaggedTrees<Sentence, Boolean> classifier) {
		StringBuilder vectorLineBuilder = new StringBuilder();
		TreeMap<Integer, Object> libLinearFeatureVector = new TreeMap<Integer, Object>();
		LibLinearFeatureManager manager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

		List<String> words = sentence.words();

		// Creating the feature vectors
		for (LibLinearFeatureManager.LibLinearFeature feature : LibLinearFeatureManager.LibLinearFeature.values())
		{
			int id;

			switch (feature)
			{
				case CONTAINS_UNIGRAM:
					for (String w : words)
					{
						id = manager.getIdFor(feature, w);
						libLinearFeatureVector.put(id, true);
					}
					break;
				case CONTAINS_BIGRAM:
					String bigram;

					for (int i = 0; i < words.size() + 1; i++)
					{
						if (i == 0)
						{
							bigram = PHI_WORD + " " + words.get(i);
						} else if (i == words.size())
						{
							bigram = words.get(i - 1) + " " + OMEGA_WORD;
						} else
						{
							bigram = words.get(i - 1) + " " + words.get(i);
						}

						id = manager.getIdFor(feature, bigram);
						libLinearFeatureVector.put(id, true);
					}

					break;
				case OBJECTIVITY_OF_SENTENCE:
					id = manager.getIdFor(feature, "");
					int objectivity = 0;

					for (String w : words)
					{
						objectivity += sentiWordNetDictionary.getObjectivityOf(w);
					}

					objectivity /= words.size();
					libLinearFeatureVector.put(id, objectivity);

					break;

				case OBJECTIVITY_OF_RELATED_WORD:
					for (String w : words) {
						if (!relatedWordsMap.containsKey(w.toLowerCase())) {
							relatedWordsMap.put(w.toLowerCase(), getWordsRelatedTo(w.toLowerCase()));
						}

						for (String relatedWord : relatedWordsMap.get(w.toLowerCase())) {
							id = manager.getIdFor(LibLinearFeatureManager.LibLinearFeature.HAS_WORD_RELATED_TO_OTHER_WORD, relatedWord);
							libLinearFeatureVector.put(id, true);

							objectivity = sentiWordNetDictionary.getObjectivityOf(relatedWord);
							id = manager.getIdFor(LibLinearFeatureManager.LibLinearFeature.OBJECTIVITY_OF_RELATED_WORD, "");
							libLinearFeatureVector.put(id, objectivity);
						}
					}
					break;
				case HAS_NAMED_ENTITY:
					for (String entity : sentence.nerTags()) {
						id = manager.getIdFor(feature, entity);
						libLinearFeatureVector.put(id, true);
					}
					break;
				case BAGGED_TREE_VOTER:
					ArrayList<NewsArticle> tmp = new ArrayList<NewsArticle>();
					tmp.add(article);
					LearnerExample<Sentence, Boolean> example = getOpinionatedSentenceExamples(tmp).get(0);
					List<Boolean> guesses = classifier.allGuessesFor(example);

					for (int i = 0; i < guesses.size(); i++) {
						id = manager.getIdFor(feature, i);
						boolean g = guesses.get(i);

						if (g)
							libLinearFeatureVector.put(id, 1);
						else
							libLinearFeatureVector.put(id, 1);
					}
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

		return vectorLineBuilder.toString();
	}

	///////////////////////
	// TRAIN CLASSIFIERS //
	///////////////////////

	private static void trainLibLinear(String vectorFileName, String modelFileName) {
		// Now that the vector file is put together, we need to run liblinear
		try
		{
			Runtime.getRuntime().exec("./liblinear_train " + vectorFileName + " " + modelFileName);
			Thread.sleep(300); // Because sometimes liblinear doesn't print out to the model file in time
		} catch (IOException e)
		{
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	//////////////////////
	// TEST CLASSIFIERS //
	//////////////////////

	//////////////////////////////////////
	// EXTRACT/EVALUATE WITH CLASSIFIER //
	//////////////////////////////////////

	/**
	 * Returns the accuracy of the classifier provided using the examples provided. Make sure that the generic types
	 * match; this method throws an UncheckedCastException if they don't match.
	 *
	 * @param examples -
	 * @param classifier -
	 * @return double
	 */
	private static double accuracyOfClassifier(List<LearnerExample> examples, BaggedTrees classifier) {
		double correct = 0.0;
		for (LearnerExample e : examples)
			if (classifier.guessFor(e).equals(e.getLabel()))
				correct += 1.0;

		return correct / (double) examples.size();
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

	private static void getRelatedWordsMap() {
		System.out.println("Gathering related words...");
		File file = new File(RELATED_WORDS_FILE);
		Gson gson = new Gson();

		if(file.exists()) {
			StringBuilder builder = new StringBuilder();

			try {
				Scanner s = new Scanner(file);
				while (s.hasNextLine()) builder.append(s.nextLine());
				s.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();

			}

			relatedWordsMap = gson.fromJson(builder.toString(), new TypeToken<TreeMap<String, String[]>>(){}.getType());
		} else {
			System.out.println("Generating new related words file! This is going to take a while...");

			// We need to gather everything and create the file
			ArrayList<NewsArticle> articles = new ArrayList<NewsArticle>();
			articles.addAll(getAllDocsFrom(DEV_DOCS));
			articles.addAll(getAllDocsFrom(TEST_DOCS));

			relatedWordsMap = new TreeMap<String, String[]>();

			for (NewsArticle a : articles) {
				Document d = new Document(a.getFullText());

				for (Sentence s : d.sentences()) {
					for (String w : s.words()) {
						if (!relatedWordsMap.containsKey(w.toLowerCase())) {
							relatedWordsMap.put(w.toLowerCase(), getWordsRelatedTo(w.toLowerCase()));
						}
					}
				}
			}

			System.out.print("Creating file...");
			try {
				PrintWriter printWriter = new PrintWriter(file);
				printWriter.print(gson.toJson(relatedWordsMap));
				printWriter.flush();
				printWriter.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}

			System.out.println("done.");
		}
	}

	/**
	 * Returns an array of words related to the word provided, using the DataMuse API.
	 *
	 * @param word -
	 * @return array of String
	 */
	private static String[] getWordsRelatedTo(String word) {
		DataMuseWord[] dataMuseWords = DataMuse.getWordsRelatedTo(word.toLowerCase());
		String[] relatedWords = new String[dataMuseWords.length];

		for (int i = 0; i < relatedWords.length; i++)
			relatedWords[i] = dataMuseWords[i].word;

		return relatedWords;
	}

	/**
	 * Sets up the learner feature manager, either pulling it from the file or instantiating it.
	 */
	private static void getLearnerFeatureManager() {
		ArrayList<NewsArticle> devArticles = getAllDocsFrom(DEV_DOCS);
		LearnerFeatureManager manager = LearnerFeatureManager.getInstance(LEARNER_FEATURE_MANAGER_FILE);

		for (NewsArticle a : devArticles) {
			Document document = new Document(a.getFullText());

			for (Sentence s : document.sentences()) {
				for (LearnerFeature f : LearnerFeature.values()) {
					switch(f) {
						case THIS_WORD:
						case CONTAINS_UNIGRAM:
							for (String w : s.words())
								manager.getIdFor(f, w);
							break;
						case CONTAINS_BIGRAM:
							for (int i = 0; i <= s.words().size(); i++) {
								String bigram;

								if (i == 0)
									bigram = PHI_WORD + " " + s.word(i);
								else if (i == s.words().size())
									bigram = s.word(i - 1) + " " + OMEGA_WORD;
								else
									bigram = s.word(i - 1) + " " + s.word(i);

								manager.getIdFor(f, bigram);
							}
							break;
						case THIS_POS:
							for (String pos : s.posTags())
								manager.getIdFor(f, pos);
							break;
						case OBJECTIVITY_OF_WORD:
							for (int i = 0; i <= 100; i++)
								manager.getIdFor(f, i);
							break;
						default:
							throw new RuntimeException("Did not include way to get IDs for " + f);
					}
				}
			}
		}

		// TODO: Anything else that needs to be here
	}

	public static Set<String> getAllWords() {
		return allWords;
	}

	public static Set<String> getAllPos() {
		return allPos;
	}
}
