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
	// VARIABLES //
	//////////////////////

	public static final String TOPTUR_DATA_FOLDER = ".toptur_data/";

	public static final String DEV_DOCS = "dataset/dev";
	public static final String TEST_DOCS = "dataset/test";
	public static final String ORIGINAL_DOCS = "dataset/original_dataset/docs";

	public static final String AGENT_TRAINING_FILE = TOPTUR_DATA_FOLDER + "agent_train.vector";
	public static final String AGENT_MODEL_FILE = TOPTUR_DATA_FOLDER + "liblinear_models/agent.model";
	public static final String AGENT_TEST_FILE = TOPTUR_DATA_FOLDER + "agent_test.vector";

	public static final String TARGET_TRAINING_FILE = TOPTUR_DATA_FOLDER + "target_train.vector";
	public static final String TARGET_MODEL_FILE = TOPTUR_DATA_FOLDER + "liblinear_models/target.model";
	public static final String TARGET_TEST_FILE = TOPTUR_DATA_FOLDER + "target_test.vector";

	public static final String LIB_LINEAR_FEATURE_MANAGER_FILE = TOPTUR_DATA_FOLDER + "lib_linear_feature_manager.json";
	public static final String SENTI_WORD_NET_FILE = "sentiwordnet.txt";
	public static final String RELATED_WORDS_FILE = TOPTUR_DATA_FOLDER + "related_words.json";
	public static final String LEARNER_FEATURE_MANAGER_FILE = TOPTUR_DATA_FOLDER + "learner_feature_manager.json";

	public static final String BAGGED_TREES_SENTENCE_CLASSIFIER = TOPTUR_DATA_FOLDER + "bagged_trees_sentences.json";
	public static final String BAGGED_TREES_OPINION_CLASSIFIER = TOPTUR_DATA_FOLDER + "bagged_trees_opinions.json";
	public static final String BAGGED_TREES_POLARITY_CLASSIFIER = TOPTUR_DATA_FOLDER + "bagged_trees_polarity.json";

	public static final String SENTENCES_LIB_LINEAR_MODEL_FILE = TOPTUR_DATA_FOLDER + "lib_linear_sentences.model";
	public static final String OPINIONS_LIB_LINEAR_MODEL_FILE = TOPTUR_DATA_FOLDER + "lib_linear_opinons.model";
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
	private static int NUMBER_OF_TREES = 2;
	private static int DEPTH_OF_TREES = 1;

	private static final String W_WORD = "__W_WORD__";
	private static final String W_PREV = "__W_PREV__";
	private static final String W_NEXT = "__W_NEXT__";
	private static final String W_POS = "__W_POS__";
	private static final String W_POS_PREV = "__W_POS_PREV__";
	private static final String W_POS_NEXT = "__W_POS_NEXT__";

	private static final String NULL_WORD = "__NULL_WORD__";
	private static final String NULL_PREV = "__NULL_PREV__";
	private static final String NULL_NEXT = "__NULL_NEXT__";
	private static final String NULL_POS = "__NULL_POS__";
	private static final String NULL_POS_PREV = "__NULL_POS_PREV__";
	private static final String NULL_POS_NEXT = "__NULL_POS_NEXT__";

	private static final String IMP_WORD = "__IMP_WORD__";
	private static final String IMP_PREV = "__IMP_PREV__";
	private static final String IMP_NEXT = "__IMP_NEXT__";
	private static final String IMP_POS = "__IMP_POS__";
	private static final String IMP_POS_PREV = "__IMP_POS_PREV__";
	private static final String IMP_POS_NEXT = "__IMP_POS_NEXT__";

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
			System.out.println("Training the sentence classifier!");
			System.out.print("\tbagged trees...");
			List<LearnerExample<Sentence, Boolean>> sentenceExamples = getOpinionatedSentenceExamples(devArticles);
			BaggedTrees<Sentence, Boolean> opinionatedSentenceClassifier = new BaggedTrees<Sentence, Boolean>(sentenceExamples,
					LearnerFeatureManager.getInstance(LEARNER_FEATURE_MANAGER_FILE).getIdsFor(LearnerFeature.getSentenceFeatures()), NUMBER_OF_TREES, DEPTH_OF_TREES);
			System.out.println("done.");
			System.out.print("\tliblinear...");
			createSentencesVectorFile(devArticles, ".sentences.vector", opinionatedSentenceClassifier);
			trainLibLinear(".sentences.vector", SENTENCES_LIB_LINEAR_MODEL_FILE);
			System.out.println("done.");

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

			// Train to detect opinion agents
			createTrainVectorFileAgent(devArticles, AGENT_TRAINING_FILE);
			trainLibLinear(AGENT_TRAINING_FILE, AGENT_MODEL_FILE);

			// Train to detect opinion agents
			createTrainVectorFileTarget(devArticles, TARGET_TRAINING_FILE);
			trainLibLinear(TARGET_TRAINING_FILE, TARGET_MODEL_FILE);


			System.out.println("Training the polarity classifier!");
			System.out.print("\tbagged trees...");
			List<LearnerExample<String, Integer>> polarityExamples = getPolarityExamples(devArticles);
			BaggedTrees<String, Integer> polarityClassifier = new BaggedTrees<String, Integer>(polarityExamples,
					LearnerFeatureManager.getInstance(LEARNER_FEATURE_MANAGER_FILE).getIdsFor(LearnerFeature.getPolarityPhraseFeatures()), NUMBER_OF_TREES, DEPTH_OF_TREES);
			System.out.println("done.");
			System.out.print("liblinear...");
			createPolarityVectorFile(devArticles, ".polarities.vector", polarityClassifier);
			trainLibLinear(".polarities.vector", POLARITY_LIB_LINEAR_MODEL_FILE);
			System.out.println("done.");

			System.out.println("Saving classifiers to disk...");
			opinionatedSentenceClassifier.saveToFile(BAGGED_TREES_SENTENCE_CLASSIFIER);
			opinionatedWordClassifier.saveToFile(BAGGED_TREES_OPINION_CLASSIFIER);
			polarityClassifier.saveToFile(BAGGED_TREES_POLARITY_CLASSIFIER);
			LearnerFeatureManager.getInstance(null).saveInstance(LEARNER_FEATURE_MANAGER_FILE);
			LibLinearFeatureManager.saveInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

            System.out.println("___FINISHED TRAINING___");

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

			Gson gson = new Gson();

			// Grab the sentence classifier
			BaggedTrees<Sentence, Boolean> sentenceClassifier = null;
			try {
				StringBuilder builder = new StringBuilder();
				Scanner s = new Scanner(new File(BAGGED_TREES_SENTENCE_CLASSIFIER));
				while(s.hasNextLine()) builder.append(s.nextLine());
				s.close();

				sentenceClassifier = gson.fromJson(builder.toString(), new TypeToken<BaggedTrees<Sentence, Boolean>>(){}.getType());
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				System.exit(1);
			}

			// Grab the opinion classifier
			BaggedTrees<String, Integer> opinionClassifier = null;
			try {
				StringBuilder builder = new StringBuilder();
				Scanner s = new Scanner(new File(BAGGED_TREES_OPINION_CLASSIFIER));
				while(s.hasNextLine()) builder.append(s.nextLine());
				s.close();

				opinionClassifier = gson.fromJson(builder.toString(), new TypeToken<BaggedTrees<String, Integer>>(){}.getType());
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				System.exit(1);
			}

			// Grab the polarity classifier
			BaggedTrees<String, Integer> polarityClassifier = null;
			try {
				StringBuilder builder = new StringBuilder();
				Scanner s = new Scanner(new File(BAGGED_TREES_POLARITY_CLASSIFIER));
				while(s.hasNextLine()) builder.append(s.nextLine());
				s.close();

				polarityClassifier = gson.fromJson(builder.toString(), new TypeToken<BaggedTrees<String, Integer>>(){}.getType());
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				System.exit(1);
			}

			// Setup the LibLinear stuff
			createSentencesVectorFile(testArticles, ".sentences.vector", sentenceClassifier);
			createOpinionatedPhraseVectorFile(testArticles, ".opinions.vector", opinionClassifier);
			createPolarityVectorFile(testArticles, ".polarity.vector", polarityClassifier);

			testLibLinear(".sentences.vector", SENTENCES_LIB_LINEAR_MODEL_FILE, "/dev/null");
			testLibLinear(".opinions.vector", OPINIONS_LIB_LINEAR_MODEL_FILE, "/dev/null");
			testLibLinear(".polarity.vector", POLARITY_LIB_LINEAR_MODEL_FILE, "/dev/null");

			createTrainVectorFileAgent(testArticles, AGENT_TEST_FILE);
			createTrainVectorFileTarget(testArticles, TARGET_TEST_FILE);

			testLibLinear(AGENT_TEST_FILE, AGENT_MODEL_FILE, "/dev/null");
			testLibLinear(TARGET_TEST_FILE, TARGET_MODEL_FILE, "/dev/null");

			// Extract the opinions
			// Let's time how long it takes

			if (evalOptions.size() == 1 && evalOptions.contains(EvaluationOption.TARGET))
				evaluateTargetClassifier(testArticles);
			else if (evalOptions.size() == 1 && evalOptions.contains(EvaluationOption.AGENT))
				evaluateAgentClassifier(testArticles);
			else
			{
				System.out.println("Starting extraction...");
				long startTime = System.currentTimeMillis();
				for (NewsArticle a : testArticles)
					extractOpinionFramesFor(a, sentenceClassifier, opinionClassifier, polarityClassifier);
				long endTime = System.currentTimeMillis();

				System.out.println(((double) endTime - startTime) / 1000.0 + " seconds");
				evaluateExtractedOpinions(testArticles, evalOptions);
			}

			evaluateExtractedOpinions(testArticles, evalOptions);

		} else if (task.equals("extract"))
		{
			Gson gson = new Gson();

			// Grab the sentence classifier
			BaggedTrees<Sentence, Boolean> sentenceClassifier = null;
			try {
				StringBuilder builder = new StringBuilder();
				Scanner s = new Scanner(new File(BAGGED_TREES_SENTENCE_CLASSIFIER));
				while(s.hasNextLine()) builder.append(s.nextLine());
				s.close();

				sentenceClassifier = gson.fromJson(builder.toString(), new TypeToken<BaggedTrees<Sentence, Boolean>>(){}.getType());
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				System.exit(1);
			}

			// Grab the opinion classifier
			BaggedTrees<String, Integer> opinionClassifier = null;
			try {
				StringBuilder builder = new StringBuilder();
				Scanner s = new Scanner(new File(BAGGED_TREES_OPINION_CLASSIFIER));
				while(s.hasNextLine()) builder.append(s.nextLine());
				s.close();

				opinionClassifier = gson.fromJson(builder.toString(), new TypeToken<BaggedTrees<String, Integer>>(){}.getType());
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				System.exit(1);
			}

			// Grab the polarity classifier
			BaggedTrees<String, Integer> polarityClassifier = null;
			try {
				StringBuilder builder = new StringBuilder();
				Scanner s = new Scanner(new File(BAGGED_TREES_POLARITY_CLASSIFIER));
				while(s.hasNextLine()) builder.append(s.nextLine());
				s.close();

				polarityClassifier = gson.fromJson(builder.toString(), new TypeToken<BaggedTrees<String, Integer>>(){}.getType());
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				System.exit(1);
			}

			for (int i = 1; i < args.length; i++)
			{
				File file = new File(args[i]);

				if (file.exists())
				{
					NewsArticle article = new NewsArticle(file);
					extractOpinionFramesFor(article, sentenceClassifier, opinionClassifier, polarityClassifier);
					System.out.println(article);
				} else
				{
					System.err.println("\"" + args[i] + "\" could not be found!");
				}
			}
		} else if (task.equals("explore"))
		{
			ArrayList<NewsArticle> devArticles = getAllDocsFrom(DEV_DOCS);
			ArrayList<NewsArticle> testArticles = getAllDocsFrom(TEST_DOCS);
			devArticles.addAll(testArticles);
			
			String test = "testimony of the confidence and high esteem";
			
			for (int i = 0; i < 7; i++)
			{
				System.out.println(getText(test, i));
				System.out.println(getPos(test, i));
			}

			int numOpinion = 0;
			int numW = 0;
			int numImp = 0;
			int numNull = 0;
			int numWord = 0;
			int numTW = 0;
			int numTNull = 0;
			int numTWord = 0;

			for (NewsArticle article : devArticles)
			{
				HashMap<String, Opinion> gold = article.getGoldStandardOpinions();
				for (String key : gold.keySet())
				{
					numOpinion++;
					if (gold.get(key).agent == null)
					{
						numNull++;
					} else if (gold.get(key).agent.equals("w"))
					{
						numW++;
					} else if (gold.get(key).agent.equals("implicit"))
					{
						numImp++;
					} else
					{
						numWord++;
					}

					if (gold.get(key).target == null)
					{
						numTNull++;
					} else if (gold.get(key).target.equals("w"))
					{
						numTW++;
					} else
					{
						numTWord++;
					}
				}
			}
			System.out.println("Number of Opinions: " + numOpinion + "\n");
			System.out.println("Number of Null Agents: " + numNull + "\n");
			System.out.println("Number of W Agents: " + numW + "\n");
			System.out.println("Number of Implicit Agents: " + numImp + "\n");
			System.out.println("Number of Word Agents: " + numWord + "\n");
			System.out.println("Number of Null Targets: " + numTNull + "\n");
			System.out.println("Number of W Targets: " + numTW + "\n");
			System.out.println("Number of Word Targets: " + numTWord + "\n");
		} else
		{
			System.err.println("No valid task provided");
			System.exit(1);
		}
	}

	/**
	 * Creates a vector file for the polarity of a given sentence.
	 *
	 * @param articles
	 *            - articles to create file from
	 * @param vectorFileName
	 *            - name of the file
	 */
	private static void createPolarityVectorFile(List<NewsArticle> articles, String vectorFileName)
	{
		StringBuilder vectorFileBuilder = new StringBuilder();

		for (NewsArticle a : articles)
		{
			for (Opinion o : a.getGoldStandardOpinions().values())
			{
				vectorFileBuilder.append(o.sentimentId());
				vectorFileBuilder.append(' ');

				Sentence sentence = new Sentence(o.sentence);
				String line = generateSentenceLineVectorFileString(null, sentence);
				vectorFileBuilder.append(line);
				vectorFileBuilder.append('\n');
			}
		}

		try
		{
			PrintWriter vectorFile = new PrintWriter(vectorFileName);
			vectorFile.print(vectorFileBuilder.toString());
			vectorFile.flush();
			vectorFile.close();
		} catch (FileNotFoundException e)
		{
			e.printStackTrace();
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
	 *            -
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

	/**
	 * Returns a list of polarity examples from the articles provided
	 * @param articles -
	 * @return -
	 */
	private static List<LearnerExample<String, Integer>> getPolarityExamples(List<NewsArticle> articles) {
		List<LearnerExample<String, Integer>> examples = new ArrayList<LearnerExample<String, Integer>>();
		LearnerExampleValueRequestListener<String> listener = new LearnerExampleValueRequestListener<String>() {
			public Object valueOfFeatureForExample(String example, int featureId) { // Sends in a String of the sentence
				LearnerFeature f = LearnerFeatureManager.getInstance(LEARNER_FEATURE_MANAGER_FILE).getLearnerFeatureFor(featureId);
				Object val = LearnerFeatureManager.getInstance(LEARNER_FEATURE_MANAGER_FILE).getValueFor(featureId);
				Sentence sentence = new Sentence(example);

				switch(f) {
					case CONTAINS_UNIGRAM:
						String unigram = (String) val;
						return sentence.words().contains(unigram);
					case CONTAINS_BIGRAM:
						String bigram = (String) val;

						for (int i = 0; i <= sentence.words().size(); i++) {
							String tmp;

							if (i == 0)
								tmp = PHI_WORD + " " + sentence.word(i);
							else if (i == sentence.words().size())
								tmp = sentence.word(i - 1) + " " + OMEGA_WORD;
							else
								tmp = sentence.word(i - 1) + " " + sentence.word(i);

							if (tmp.equalsIgnoreCase(bigram))
								return true;
						}

						return false;
					default:
						throw new RuntimeException("Could not return something for polarity");
				}
			}
		};

		for (NewsArticle a : articles) {
			for (Opinion o : a.getGoldStandardOpinions().values()) {
				examples.add(new LearnerExample<String, Integer>(o.sentence, o.sentimentId(), listener));
			}
		}

		return examples;
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
	 *            -
	 * @param nameOfVectorFile
	 *            -
	 */
	private static void createSentencesVectorFile(List<NewsArticle> articles, String nameOfVectorFile, BaggedTrees<Sentence, Boolean> classifier) {
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
									id = manager.getIdFor(feature, Integer.toString(j)); // NOTE: Using strings because integers aren't okay for some reason
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
	 * TODO: This looks important.
	 */
	private static String createSingleOpinionAgentTarget(Opinion opinion, ArrayList<Pair<String, String>> wordPos, ArrayList<String> words, int index)
	{
		LibLinearFeatureManager manager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

		String word;
		String pos;

		if (words.get(index).equals(W_WORD))
		{
			word = W_WORD;
			pos = W_POS;
		} else if (words.get(index).equals(NULL_WORD))
		{
			word = NULL_WORD;
			pos = NULL_POS;
		} else if (words.get(index).equals(IMP_WORD))
		{
			word = IMP_WORD;
			pos = IMP_POS;
		} else
		{
			word = words.get(index);
			pos = wordPos.get(index).getSecond();
		}

		StringBuilder vectorLineBuilder = new StringBuilder();
		// vectorLineBuilder.append(0);
		// vectorLineBuilder.append(' ');
		TreeMap<Integer, String> featureVector = new TreeMap<Integer, String>();

		// Creating the feature vectors
		for (LibLinearFeatureManager.LibLinearFeature feature : LibLinearFeatureManager.LibLinearFeature.values())
		{
			int id;

			switch (feature)
			{
				case BAG_OF_WORDS:
					for (String w: words)
					{
						id = manager.getIdFor(feature, w);
						featureVector.put(id, id + ":1");
					}
					break;
				case PREVIOUS_UNIGRAM:
					String previous;

					if (word.equals(W_WORD))
						previous = W_PREV;
					else if (word.equals(NULL_WORD))
						previous = NULL_PREV;
					else if (word.equals(IMP_WORD))
						previous = IMP_PREV;
					else if (index > 0)
						previous = words.get(index-1);
					else
						previous = PHI_WORD;

					id = manager.getIdFor(feature, previous);
					featureVector.put(id, id + ":1");
					break;
				case THIS_UNIGRAM:
					id = manager.getIdFor(feature, word);
					featureVector.put(id, id + ":1");
					break;
				case NEXT_UNIGRAM:
					String next;

					if (word.equals(W_WORD))
						next = W_NEXT;
					else if (word.equals(NULL_WORD))
						next = NULL_NEXT;
					else if (word.equals(IMP_WORD))
						next = IMP_NEXT;
					else if (index < wordPos.size() - 1)
						next = words.get(index+1);
					else
						next = OMEGA_WORD;

					id = manager.getIdFor(feature, next);
					featureVector.put(id, id + ":1");
					break;
				case PREVIOUS_PART_OF_SPEECH:
					String previousPos;

					if (word.equals(W_WORD))
						previousPos = W_POS_PREV;
					else if (word.equals(NULL_WORD))
						previousPos = NULL_POS_PREV;
					else if (word.equals(IMP_WORD))
						previousPos = IMP_POS_PREV;
					else if (index > 0)
						previousPos = wordPos.get(index-1).getSecond();
					else
						previousPos = PHI_POS;

					id = manager.getIdFor(feature, previousPos);
					featureVector.put(id, id + ":1");
					break;
				case THIS_PART_OF_SPEECH:
					id = manager.getIdFor(feature, pos);
					featureVector.put(id, id + ":1");
					break;
				case NEXT_PART_OF_SPEECH:
					String nextPos;

					if (word.equals(W_WORD))
						nextPos = W_POS_NEXT;
					else if (word.equals(NULL_WORD))
						nextPos = NULL_POS_NEXT;
					else if (word.equals(IMP_WORD))
						nextPos = IMP_POS_NEXT;
					else if (index < wordPos.size() - 1)
						nextPos = wordPos.get(index+1).getSecond();
					else
						nextPos = OMEGA_POS;

					id = manager.getIdFor(feature, nextPos);
					featureVector.put(id, id + ":1");
					break;
				case OBJECTIVITY_OF_WORD:
					id = manager.getIdFor(feature, "");
					double objectivity = (double) sentiWordNetDictionary.getObjectivityOf(word);
					objectivity = objectivity / 100.0;
					featureVector.put(id, id + ":" + objectivity);
					break;
				case POS_DISTRIBUTION:
					ArrayList<Sentence> sentences = (ArrayList<Sentence>) new Document(opinion.sentence).sentences();
					HashMap<String, Double> posFreq = new HashMap<String, Double>();
					int total = 0;
					for(Sentence s: sentences)
					{
						for (int i = 0; i < s.length(); i++)
						{
							String posTag = s.posTag(i);
							if(posFreq.containsKey(posTag))
							{
								posFreq.put(posTag, posFreq.get(posTag) + 1);
							}
							else
							{
								posFreq.put(posTag, 1.0);
							}
							total++;
						}
					}

					for(String key: posFreq.keySet()) {
						double freq = posFreq.get(key) / total;
						id = manager.getIdFor(feature, key);
						featureVector.put(id, id + ":" + freq);
					}
					break;
				case IS_ENTITY:
					ArrayList<Sentence> sentences2 = (ArrayList<Sentence>) new Document(opinion.sentence).sentences();
					String ner = "";
					for(Sentence s: sentences2)
					{
						for(int i = 0; i < s.length(); i++)
						{
							if(s.word(i).equals(words.get(index)))
							{
								ner = s.nerTag(i);
							}
						}
					}
					id = manager.getIdFor(feature, ner);
					featureVector.put(id, id + ":1");
					break;
				case NUM_ENTITIES:
					ArrayList<Sentence> sentences3 = (ArrayList<Sentence>) new Document(opinion.sentence).sentences();
					int nerCount = 0;
					for(Sentence s: sentences3)
					{
						for(int i = 0; i < s.length(); i++)
						{
							if(!s.nerTag(i).equals("O"))
							{
								nerCount++;
							}
						}
					}
					id = manager.getIdFor(feature, "");
					featureVector.put(id, id + ":" + nerCount);
					break;
				case LEN_SENT:
					id = manager.getIdFor(feature, "");
					featureVector.put(id, id + ":" + opinion.sentence.split("\\s+").length);
					break;
				case LEN_OP:
					id = manager.getIdFor(feature, "");
					featureVector.put(id, id + ":" + wordPos.size());
					break;
			}

		}

		for (String s : featureVector.values())
		{
			vectorLineBuilder.append(s);
			vectorLineBuilder.append(' ');
		}

		return vectorLineBuilder.toString();

	}


	private static void createTrainVectorFileAgent(ArrayList<NewsArticle> articles, String nameOfVectorFile)
	{
		StringBuilder vectorFileBuilder = new StringBuilder();
		LibLinearFeatureManager manager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

		for (NewsArticle article : articles)
		{
			for (String opinion_phrase : article.getGoldStandardOpinions().keySet())
			{
				Opinion opinion_record = article.getGoldStandardOpinions().get(opinion_phrase);

				ArrayList<Pair<String, String>> words = processSentence(opinion_phrase);

				String vec_file = createVectorFileAgent(opinion_record, opinion_phrase, words);

				vectorFileBuilder.append(vec_file);
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
	 * Generates a vector file in LibLinear format for whatever articles are
	 * provided.
	 *
	 * This vector file contains features for identifying the Agent of an opinion.
	 *
	 * @param opinion
	 * @param opinion_phrase
	 * @param wordPos
	 */
	private static String createVectorFileAgent(Opinion opinion, String opinion_phrase, ArrayList<Pair<String, String>> wordPos)
	{
		StringBuilder vectorFileBuilder = new StringBuilder();

		ArrayList<String> words = new ArrayList<String>();

		for(Pair<String, String> p: wordPos)
		{
			words.add(p.getFirst());
		}

		String agent;

		if (opinion.agent == null)
		{
			agent = "null";
		} else
		{
			agent = opinion.agent;
		}

		words.add(W_WORD);
		words.add(NULL_WORD);
		words.add(IMP_WORD);

		for (int i = 0; i < words.size(); i++)
		{
			String word;

			if (words.get(i).equals(W_WORD))
			{
				word = W_WORD;
			} else if (words.get(i).equals(NULL_WORD))
			{
				word = NULL_WORD;
			} else if (words.get(i).equals(IMP_WORD))
			{
				word = IMP_WORD;
			} else
			{
				word = words.get(i);
			}

			StringBuilder vectorLineBuilder = new StringBuilder();

			if (word.equals(W_WORD) && agent.equals("w"))
			{
				vectorLineBuilder.append("1 ");
			} else if (word.equals(NULL_WORD) && agent.equals("null"))
			{
				vectorLineBuilder.append("1 ");
			} else if (word.equals(IMP_WORD) && agent.equals("implicit"))
			{
				vectorLineBuilder.append("1 ");
			} else if (word.equalsIgnoreCase(agent))
			{
				vectorLineBuilder.append("1 ");
			} else
			{
				vectorLineBuilder.append("0 ");
			}

			vectorLineBuilder.append(createSingleOpinionAgentTarget(opinion, wordPos, words, i));

			// Creating the feature vectors

			vectorFileBuilder.append(vectorLineBuilder.toString());
			vectorFileBuilder.append("\n");
		}

		return vectorFileBuilder.toString();

	}

	private static void createTrainVectorFileTarget(ArrayList<NewsArticle> articles, String nameOfVectorFile)
	{
		StringBuilder vectorFileBuilder = new StringBuilder();

		for (NewsArticle article : articles)
		{
			for (String opinion_phrase : article.getGoldStandardOpinions().keySet())
			{
				Opinion opinion_record = article.getGoldStandardOpinions().get(opinion_phrase);

				ArrayList<Pair<String, String>> words = processSentence(opinion_phrase);

				String vec_file = (createVectorFileTarget(opinion_record, opinion_phrase, words));

				vectorFileBuilder.append(vec_file);
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
	 * Generates a vector file in LibLinear format for whatever articles are
	 * provided.
	 *
	 * This vector file contains features for identifying the Target of an opinion.
	 *
	 * @param wordPos
	 * @param opinionPhrase
	 * @param wordPos
	 * @param wordPos
	 */
	private static String createVectorFileTarget(Opinion opinion, String opinionPhrase, ArrayList<Pair<String, String>> wordPos)
	{
		StringBuilder vectorFileBuilder = new StringBuilder();

		ArrayList<String> words = new ArrayList<String>();

		for(Pair<String, String> p: wordPos)
		{
			words.add(p.getFirst());
		}

		String target;

		if (opinion.target == null)
		{
			target = "null";
		} else
		{
			target = opinion.target;
		}

		words.add(NULL_WORD);

		for (int i = 0; i < words.size(); i++)
		{
			String word;

			if (words.get(i).equals(NULL_WORD))
			{
				word = NULL_WORD;
			} else
			{
				word = words.get(i);
			}

			StringBuilder vectorLineBuilder = new StringBuilder();

			if (word.equals(NULL_WORD) && target.equals("null"))
			{
				vectorLineBuilder.append("1 ");
			} else if (word.equalsIgnoreCase(target))
			{
				vectorLineBuilder.append("1 ");
			} else
			{
				vectorLineBuilder.append("0 ");
			}

			vectorLineBuilder.append(createSingleOpinionAgentTarget(opinion, wordPos, words, i));

			vectorFileBuilder.append(vectorLineBuilder.toString());
			vectorFileBuilder.append("\n");
		}

		return vectorFileBuilder.toString();
	}

	/**
	 * Creates a vector for polarity.
	 *
	 * @param articles -
	 * @param fileName -
	 * @param classifier -
	 */
	private static void createPolarityVectorFile(ArrayList<NewsArticle> articles, String fileName, BaggedTrees<String, Integer> classifier) {
		StringBuilder vectorLineBuilder = new StringBuilder();
		TreeMap<Integer, Object> libLinearFeatureVector = new TreeMap<Integer, Object>();
		LibLinearFeatureManager manager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

		for (NewsArticle a : articles) {
			for (Opinion o : a.getGoldStandardOpinions().values()) {
				Sentence sentence = new Sentence(o.sentence);
				List<String> words = sentence.words();

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
							tmp.add(a);
							LearnerExample<String, Integer> example = getPolarityExamples(tmp).get(0);
							List<Integer> guesses = classifier.allGuessesFor(example);

							for (int i = 0; i < guesses.size(); i++) {
								id = manager.getIdFor(feature, Integer.toString(i));
								int g = guesses.get(i);
								libLinearFeatureVector.put(id, g);
							}
							break;
					}
				}

				vectorLineBuilder.append(o.sentimentId());

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

				vectorLineBuilder.append("\n");
			}
		}

		try
		{
			PrintWriter vectorFile = new PrintWriter(fileName);
			vectorFile.print(vectorLineBuilder.toString());
			vectorFile.flush();
			vectorFile.close();
		} catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}
	}

		/**
		 *
		 * TODO: What is this?
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
						id = manager.getIdFor(feature, Integer.toString(i)); // NOTE: I got a weird exception here so I'm trying to throw Strings at it instead of integers
						boolean g = guesses.get(i);

						if (g)
							libLinearFeatureVector.put(id, 1);
						else
							libLinearFeatureVector.put(id, 0);
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

	private static void trainLibLinear(String vectorFileName, String modelFileName)
	{
		// Now that the vector file is put together, we need to run liblinear
		try
		{
			Process p = Runtime.getRuntime()
					.exec("./liblinear_train -B 1 -s 6 " + vectorFileName + " " + modelFileName);
			p.waitFor();
			// Because sometimes liblinear doesn't print out to the
			// model file in time
		} catch (IOException e)
		{
			e.printStackTrace();
		} catch (InterruptedException e)
		{
			e.printStackTrace();
		}
	}

	//////////////////////
	// TEST CLASSIFIERS //
	//////////////////////

	private static void testLibLinear(String testVectorFileName, String modelFileName, String outputFileName)
	{
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
		} catch (InterruptedException e)
		{
			e.printStackTrace();
		}
	}

	//////////////////////////////////////
	// EXTRACT/EVALUATE WITH CLASSIFIER //
	//////////////////////////////////////

	/**
	 * Extracts all the opinion frames for the article provided.
	 *
	 * @param a -
	 */
	private static void extractOpinionFramesFor(NewsArticle a, BaggedTrees<Sentence, Boolean> sentenceClassifier, BaggedTrees<String, Integer> opinionClassifier, BaggedTrees<String, Integer> polarityClassifier) {
		Document document = new Document(a.getFullText());

		for (Sentence s : document.sentences()) {
			if (sentenceIsOpinionated(s, sentenceClassifier)) {
				for (String expression : extractOpinionsFromSentence(s, opinionClassifier)) {
					Opinion o = new Opinion();
					o.opinion = expression;
					o.sentence = s.toString();
					o.sentiment = extractPolarityOfSentence(s, polarityClassifier);

					HashMap<String, Double> confidences = extractAgentFrom(o);

					double max = confidences.get(NULL_WORD);
					String max_word = NULL_WORD;
					for (String word : confidences.keySet())
					{
						if (confidences.get(word) > max)
						{
							max = confidences.get(word);
							max_word = word;
						}
					}

					String classifier_result;
					if (max_word.equals(W_WORD))
						classifier_result = "w";
					else if (max_word.equals(NULL_WORD))
						classifier_result = "null";
					else
						classifier_result = max_word;

					o.opinion = classifier_result;

					HashMap<String, Double> confidences2 = extractTargetFrom(o);

					double max2 = confidences2.get(NULL_WORD);
					String max_word2 = NULL_WORD;
					for (String word : confidences2.keySet())
					{
						if (confidences2.get(word) > max2)
						{
							max2 = confidences2.get(word);
							max_word2 = word;
						}
					}

					String classifier_result2;
					if (max_word2.equals(W_WORD))
						classifier_result2 = "w";
					else if (max_word2.equals(NULL_WORD))
						classifier_result2 = "null";
					else
						classifier_result2 = max_word2;

					o.target = classifier_result2;

					a.addExtractedOpinion(o);
				}
			}
		}
	}

	/**
	 * Returns true if the sentence provided is opinionated
	 *
	 * @param s -
	 * @param classifier -
	 * @return boolean
	 */
	private static boolean sentenceIsOpinionated(Sentence s, BaggedTrees<Sentence, Boolean> classifier) {
		NewsArticle tmpArticle = new NewsArticle("tmp", s.toString(), new Opinion[0]);
		List<NewsArticle> list = new ArrayList<NewsArticle>();
		list.add(tmpArticle);
		createSentencesVectorFile(list, ".sentence_tmp.vector", classifier);
		try
		{
			Process p = Runtime.getRuntime().exec("./liblinear_predict .sentence_tmp.vector " + SENTENCES_LIB_LINEAR_MODEL_FILE + " output_sentence.txt");
			p.waitFor();
			Scanner derp = new Scanner(Runtime.getRuntime().exec("cat output.txt").getInputStream());
			int i = -1;
			if (derp.hasNextInt())
			{
				i = derp.nextInt();
			}
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

	/**
	 * Returns collection of opinions from some sentence s.
	 *
	 * @param sentence -
	 * @param classifier -
	 * @return List. Yah brah.
	 */
	private static List<String> extractOpinionsFromSentence(Sentence sentence, BaggedTrees<String, Integer> classifier) {
		ArrayList<String> list = new ArrayList<String>();
		LibLinearFeatureManager manager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);
		LinkedList<Integer> bioLabels = new LinkedList<Integer>();

		// Grab the BIO labels for each word in the sentence
		for (int i = 0; i < sentence.words().size(); i++) {
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
						Opinion o = new Opinion();
						o.sentence = sentence.toString();
						o.opinion = sentence.toString();

						Opinion[] opinions = new Opinion[] { o };
						tmp.add(new NewsArticle("", sentence.toString(), opinions));
						LearnerExample<String, Integer> example = getOpinionWordExamples(tmp).get(0);
						List<Integer> guesses = classifier.allGuessesFor(example);

						for (int j = 0; j < guesses.size(); j++) {
							id = manager.getIdFor(feature, Integer.toString(j)); // NOTE: Using strings because integers aren't okay for some reason
							int g = guesses.get(j);
							stupidMap.put(id, id + ":" + g);
						}
						break;
				}
			}

			StringBuilder vectorFileBuilder = new StringBuilder();
			vectorFileBuilder.append("0 ");

			for (String s : stupidMap.values()) {
				vectorFileBuilder.append(s);
				vectorFileBuilder.append(' ');
			}

			String filename = ".opinions_tmp.vector";

			try
			{
				PrintWriter vectorFile = new PrintWriter(filename);
				vectorFile.print(vectorFileBuilder.toString());
				vectorFile.flush();
				vectorFile.close();
			} catch (FileNotFoundException e)
			{
				e.printStackTrace();
			}

			try
			{
				Process p = Runtime.getRuntime().exec("./liblinear_predict " + filename + " " + OPINIONS_LIB_LINEAR_MODEL_FILE + " output_opinion.txt");
				p.waitFor();
				Scanner derp = new Scanner(Runtime.getRuntime().exec("cat output_opinion.txt").getInputStream());

				int val = 0;
				if (derp.hasNextInt())
				{
					val = derp.nextInt();
				}
				derp.close();

				bioLabels.add(val);
			} catch (IOException e)
			{
				e.printStackTrace();
			} catch (InterruptedException e)
			{
				e.printStackTrace();
			}
		}

		// Now that we have the BIO labels, we can pull out the opinion phrases
		StringBuilder builder = new StringBuilder();
		boolean flag = false;
		for (int i = 0; i < bioLabels.size(); i++) {
			int label = bioLabels.get(i);

			if (label == 1 || label == 2) {
				flag = true;
				builder.append(sentence.word(i));
				builder.append(' ');
			} else if (flag) {
				flag = false;
				list.add(builder.toString());
				builder = new StringBuilder();
			}
		}

		return list;
	}


	/**
	 * Extracts polarity of a given sentence
	 *
	 * @param sentence
	 *            -
	 * @return String
	 */
	private static String extractPolarityOfSentence(Sentence sentence, BaggedTrees<String, Integer> classifier)
	{
		StringBuilder vectorFileBuilder = new StringBuilder();
		LibLinearFeatureManager manager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

		String vectorFileName = ".polarity_tmp.vector";

		vectorFileBuilder.append(0);
		vectorFileBuilder.append(' ');

		List<String> words = sentence.words();

		TreeMap<Integer, Object> libLinearFeatureVector = new TreeMap<Integer, Object>();
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
					Opinion o = new Opinion();
					o.sentence = sentence.toString();
					o.sentiment = "both";
					Opinion[] opinions = new Opinion[] {o};

					NewsArticle tmpArticle = new NewsArticle("", sentence.toString(), opinions);
					ArrayList<NewsArticle> tmp = new ArrayList<NewsArticle>();
					tmp.add(tmpArticle);
					LearnerExample<String, Integer> example = getPolarityExamples(tmp).get(0);
					List<Integer> guesses = classifier.allGuessesFor(example);

					for (int i = 0; i < guesses.size(); i++) {
						id = manager.getIdFor(feature, Integer.toString(i)); // NOTE: I got a weird exception here so I'm trying to throw Strings at it instead of integers
						int g = guesses.get(i);
						libLinearFeatureVector.put(id, g);
					}
					break;
			}
		}

		for (Map.Entry<Integer, Object> e : libLinearFeatureVector.entrySet())
		{
			vectorFileBuilder.append(" ");
			vectorFileBuilder.append(e.getKey());
			vectorFileBuilder.append(":");

			if (e.getValue() instanceof Boolean)
				vectorFileBuilder.append(1);
			else
				vectorFileBuilder.append(e.getValue());
		}

		vectorFileBuilder.append('\n');

		try
		{
			PrintWriter vectorFile = new PrintWriter(vectorFileName);
			vectorFile.print(vectorFileBuilder.toString());
			vectorFile.flush();
			vectorFile.close();
		} catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}

		try
		{
			Process p = Runtime.getRuntime()
					.exec("./liblinear_predict " + vectorFileName + " " + POLARITY_LIB_LINEAR_MODEL_FILE + " herp.txt");
			p.waitFor();
			Scanner s = new Scanner(new File("herp.txt"));
			int id = Integer.parseInt(s.next());
			s.close();

			return Opinion.fromSentimentId(id);
		} catch (IOException e)
		{
			e.printStackTrace();
		} catch (InterruptedException e)
		{
			e.printStackTrace();
		}

		return null;
	}

	private static HashMap<String, Double> extractAgentFrom(Opinion o) {
		ArrayList<Pair<String, String>> wordPos = processSentence(o.opinion);

		ArrayList<String> words = new ArrayList<String>();

		for (Pair<String, String> p : wordPos) {
			words.add(p.getFirst());
		}

		words.add(W_WORD);
		words.add(NULL_WORD);
		words.add(IMP_WORD);

		HashMap<String, Double> confidences = new HashMap<String, Double>();

		for (int i = 0; i < words.size(); i++) {
			String name = "some_file_" + i + ".vector";
			String features = "0 " + createSingleOpinionAgentTarget(o, wordPos, words, i);

			try {
				PrintWriter vectorFile = new PrintWriter(name);
				vectorFile.print(features);
				vectorFile.flush();
				vectorFile.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}

			try {
				Process p = Runtime.getRuntime()
						.exec("./liblinear_predict -b 1 " + name + " " + AGENT_MODEL_FILE + " output.txt");
				p.waitFor();
				String result = "";

				Scanner scanner = new Scanner("output.txt");

				while (scanner.hasNextLine()) {
					result += scanner.nextLine();
				}

				String[] tokens = result.split("\\s+");

				confidences.put(words.get(i), Double.parseDouble(tokens[4]));
				scanner.close();

			} catch (IOException e) {
				e.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}

		return confidences;
	}
		// System.out.println(confidences);
		// double max = confidences.get(W_WORD);
		// String max_word = W_WORD;
		// for (String word : words)
		// {
		// if (confidences.get(word) > max)
		// {
		// max = confidences.get(word);
		// max_word = word;
		// }
		// }
		//
		// if (max_word.equals(W_WORD))
		// return "w";
		// else if (max_word.equals(NULL_WORD))
		// return "null";
		// else
		// return max_word;
//	}

	/**
	 * TODO: Where does this go?
	 * @param o
	 * @return
	 */
	private static HashMap<String, Double> extractTargetFrom(Opinion o)
	{
		ArrayList<Pair<String, String>> wordPos = processSentence(o.opinion);

		ArrayList<String> words = new ArrayList<String>();

		for(Pair<String, String> p: wordPos)
		{
			words.add(p.getFirst());
		}
		
		words.add(NULL_WORD);

		HashMap<String, Double> confidences = new HashMap<String, Double>();

		for (int i = 0; i < words.size(); i++)
		{
			String name = "some_file_2" + i + ".vector";
			String features = "0 " + createSingleOpinionAgentTarget(o, wordPos, words, i);

			try
			{
				PrintWriter vectorFile = new PrintWriter(name);
				vectorFile.print(features);
				vectorFile.flush();
				vectorFile.close();
			} catch (FileNotFoundException e)
			{
				e.printStackTrace();
			}

			try
			{
				Process p = Runtime.getRuntime()
						.exec("./liblinear_predict -b 1 " + name + " " + TARGET_MODEL_FILE + " output.txt");
				p.waitFor(); // To give LibLinear enough time to output to file

				Scanner scanner = new Scanner(new File("output.txt"));

				String result = "";

				while (scanner.hasNextLine())
				{
					result += scanner.nextLine();
				}

				String[] tokens = result.split("\\s+");

				confidences.put(words.get(i), Double.parseDouble(tokens[4]));
			} catch (IOException e)
			{
				e.printStackTrace();
			} catch (InterruptedException e)
			{
				e.printStackTrace();
			}
		}

		return confidences;
		// System.out.println(confidences);
		// double max = confidences.get(NULL_WORD);
		// String max_word = NULL_WORD;
		// for (String word : words)
		// {
		// if (confidences.get(word) > max)
		// {
		// max = confidences.get(word);
		// max_word = word;
		// }
		// }
		//
		// if (max_word.equals(W_WORD))
		// return "w";
		// else if (max_word.equals(NULL_WORD))
		// return "null";
		// else
		// return max_word;
	}

	/**
	 * Evaluates the extracted opinions, given whatever evaluation options desired.
	 *
	 * @param articles -
	 * @param evaluationOptions -
	 */
	private static void evaluateExtractedOpinions(ArrayList<NewsArticle> articles,
												  Set<EvaluationOption> evaluationOptions)
	{
		double totalPrecision = 0.0;
		double totalRecall = 0.0;
		double totalFScore = 0.0;

		System.out.println("Name\tPrecision\tRecall\tFScore");

		for (NewsArticle article : articles)
		{
			Map<String, Opinion> extractedOpinions = (Map<String, Opinion>) article.getExtractedOpinions().clone();
			Map<String, Opinion> goldStandardOpinions = (Map<String, Opinion>) article.getGoldStandardOpinions().clone();

			double truePositives = 0.0;

			TreeSet<String> correct = new TreeSet<String>();

			for (Opinion goldStandard : goldStandardOpinions.values())
			{
				for (Opinion extracted : extractedOpinions.values())
				{
					boolean matches = true;

					for (EvaluationOption option : evaluationOptions)
					{
						if (!Opinion.opinionsMatchGivenOption(extracted, goldStandard, option))
						{
							matches = false;
							break;
						}
					}

					if (matches && !correct.contains(extracted.opinion))
					{
						correct.add(extracted.opinion);
						truePositives += 1.0;
					}
				}
			}

			for (String s : correct)
			{
				extractedOpinions.remove(s);
				goldStandardOpinions.remove(s);
			}

			double precision = truePositives / (Math.max(extractedOpinions.size() + correct.size(), 1));
			double recall = truePositives / (Math.max(goldStandardOpinions.size() + correct.size(), 1));
			double fscore = 2 * ((precision * recall) / (Math.max(1, precision + recall)));

			System.out.println(article.getDocumentName() + "\t" + precision + "\t" + recall + "\t" + fscore);

			totalPrecision += precision;
			totalRecall += recall;
			totalFScore += fscore;
		}

		System.out.println("\nTOTAL PRECISION: " + totalPrecision / articles.size());
		System.out.println("\nTOTAL RECALL: " + totalRecall / articles.size());
		System.out.println("\nTOTAL FSCORE: " + totalFScore / articles.size());
	}

	/**
	 * Returns precision, recall, and Fscore in that order.
	 *
	 * @param articles
	 */
	private static void evaluateAgentClassifier(List<NewsArticle> articles)
	{
		System.out.println("\nReport: ");

		double totalOpinions = 0.0;
		double correctLabel = 0.0; // Correct label agent or null
		double nonNullLabels = 0.0; // How Many Agent Labels
		double nonNullCorrect = 0.0; // Correct Agent Label

		Random rand = new Random();

		String samples = "";

		for (NewsArticle article : articles)
		{
			HashMap<String, Opinion> goldStandardOpinions = (HashMap<String, Opinion>) article.getGoldStandardOpinions()
					.clone();
			for (Opinion goldStandard : goldStandardOpinions.values())
			{
				String agent;
				if (goldStandard.agent == null)
				{
					agent = "null";
					totalOpinions += 1;
				} else
				{
					agent = goldStandard.agent;
					totalOpinions += 1;
					nonNullLabels += 1;
				}

				HashMap<String, Double> confidences = extractAgentFrom(goldStandard);

				double max = confidences.get(NULL_WORD);
				String max_word = NULL_WORD;
				for (String word : confidences.keySet())
				{
					if (confidences.get(word) > max)
					{
						max = confidences.get(word);
						max_word = word;
					}
				}

				String classifier_result;
				if (max_word.equals(W_WORD))
					classifier_result = "w";
				else if (max_word.equals(NULL_WORD))
					classifier_result = "null";
				else if (max_word.equals(IMP_WORD))
					classifier_result = "implicit";
				else
					classifier_result = max_word;
				
//				if (!classifier_result.equals("null"))
//				{
//					String samples2 = "";
//					samples2 += "\n\nSentence: " + goldStandard.sentence;
//					samples2 += "\nOpinion: " + goldStandard.opinion;
//					samples2 += "\nAgent: " + goldStandard.agent;
//					samples2 += "\nPredicted Agent: " + classifier_result;
//					samples2 += "\nConfidences:\n" + confidences.toString();
//					System.out.println(samples2);
//				}

				if (!agent.equals("null"))
				{
					if (classifier_result.equals(agent))
					{
						correctLabel += 1.0;
						nonNullCorrect += 1.0;
						samples += "\n\nCorrectly Identified a non-Null Agent!";
						samples += "\nOpinion: " + goldStandard.opinion;
						samples += "\nAgent: " + goldStandard.agent;
						samples += "\nPredicted Agent: " + classifier_result;
						samples += "\nConfidences:\n" + confidences.toString();
					}
				} else if (classifier_result.equals(agent))
				{
					correctLabel += 1.0;
				}

				if (rand.nextDouble() < 0.07)
				{
					samples += "\n\nSentence: " + goldStandard.sentence;
					samples += "\nOpinion: " + goldStandard.opinion;
					samples += "\nAgent: " + goldStandard.agent;
					samples += "\nPredicted Agent: " + classifier_result;
					samples += "\nConfidences:\n" + confidences.toString();
				}
			}
		}

		double totalPrecision = correctLabel / totalOpinions;
		double totalRecall = nonNullCorrect / nonNullLabels;
		double totalFScore = (2 * (totalPrecision * totalRecall)) / (totalPrecision + totalRecall);

		System.out.println("\nTOTAL OPINIONS: " + totalOpinions);
		System.out.println("TOTAL OPINIONS WITH AGENT: " + nonNullLabels);
		System.out.println("TOTAL OPINIONS LABELED CORRECTLY: " + correctLabel);
		System.out.println("TOTAL OPINIONS WITH AGENT LABELED CORRECTLY: " + nonNullCorrect);
		System.out.println("AGENT TOTAL PRECISION: " + totalPrecision);
		System.out.println("AGENT TOTAL RECALL: " + totalRecall);
		System.out.println("AGENT TOTAL FSCORE: " + totalFScore);
		System.out.println("\nSamples: ");
		System.out.println(samples + "\n");
	}

	/**
	 * Returns precision, recall, and fscore in that order
	 * 
	 * @param articles
	 * @return
	 */
	private static void evaluateTargetClassifier(List<NewsArticle> articles)
	{
		System.out.println("\nReport: ");

		double totalOpinions = 0.0;
		double correctLabel = 0.0; // Correct label agent or null
		double nonNullLabels = 0.0; // How Many Agent Labels
		double nonNullCorrect = 0.0; // Correct Agent Label

		Random rand = new Random();

		String samples = "";

		for (NewsArticle article : articles)
		{
			HashMap<String, Opinion> goldStandardOpinions = (HashMap<String, Opinion>) article.getGoldStandardOpinions()
					.clone();

			for (Opinion goldStandard : goldStandardOpinions.values())
			{
				String target;
				if (goldStandard.target == null)
				{
					target = "null";
					totalOpinions += 1;
				} else
				{
					target = goldStandard.target;
					totalOpinions += 1;
					nonNullLabels += 1;
				}

				HashMap<String, Double> confidences = extractTargetFrom(goldStandard);

				double max = confidences.get(NULL_WORD);
				String max_word = NULL_WORD;
				for (String word : confidences.keySet())
				{
					if (confidences.get(word) > max)
					{
						max = confidences.get(word);
						max_word = word;
					}
				}

				String classifier_result;
				if (max_word.equals(W_WORD))
					classifier_result = "w";
				else if (max_word.equals(NULL_WORD))
					classifier_result = "null";
				else
					classifier_result = max_word;

				if (!target.equals("null"))
				{
					if (classifier_result.equals(target))
					{
						correctLabel += 1.0;
						nonNullCorrect += 1.0;
					}
				} else if (classifier_result.equals(target))
				{
					correctLabel += 1.0;
				}

				if (rand.nextDouble() < 0.07)
				{
					samples += "\n\nSentence: " + goldStandard.sentence;
					samples += "\nOpinion: " + goldStandard.opinion;
					samples += "\nTarget: " + goldStandard.target;
					samples += "\nPredicted Target: " + classifier_result;
					samples += "\nConfidences:\n" + confidences.toString();
				}
			}

		}

		double totalPrecision = correctLabel / totalOpinions;
		double totalRecall = nonNullCorrect / nonNullLabels;
		double totalFScore = (2 * (totalPrecision * totalRecall)) / (totalPrecision + totalRecall);

		System.out.println("\nTOTAL OPINIONS: " + totalOpinions);
		System.out.println("TOTAL OPINIONS WITH TARGET: " + nonNullLabels);
		System.out.println("TOTAL OPINIONS LABELED CORRECTLY: " + correctLabel);
		System.out.println("TOTAL OPINIONS WITH TARGET LABELED CORRECTLY: " + nonNullCorrect);
		System.out.println("TARGET TOTAL PRECISION: " + totalPrecision);
		System.out.println("TARGET TOTAL RECALL: " + totalRecall);
		System.out.println("TARGET TOTAL FSCORE: " + totalFScore);
		System.out.println("\nSamples: ");
		System.out.println(samples + "\n");
	}

	////////////////////
	// HELPER METHODS //
	////////////////////

	private static void getSentiWordNet()
	{
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

	private static void getRelatedWordsMap()
	{
		System.out.println("Gathering related words...");
		File file = new File(RELATED_WORDS_FILE);
		Gson gson = new Gson();

		if (file.exists())
		{
			StringBuilder builder = new StringBuilder();

			try
			{
				Scanner s = new Scanner(file);
				while (s.hasNextLine())
					builder.append(s.nextLine());
				s.close();
			} catch (FileNotFoundException e)
			{
				e.printStackTrace();

			}

			relatedWordsMap = gson.fromJson(builder.toString(), new TypeToken<TreeMap<String, String[]>>()
			{
			}.getType());
		} else
		{
			System.out.println("Generating new related words file! This is going to take a while...");

			// We need to gather everything and create the file
			ArrayList<NewsArticle> articles = new ArrayList<NewsArticle>();
			articles.addAll(getAllDocsFrom(DEV_DOCS));
			articles.addAll(getAllDocsFrom(TEST_DOCS));

			relatedWordsMap = new TreeMap<String, String[]>();

			for (NewsArticle a : articles)
			{
				Document d = new Document(a.getFullText());

				for (Sentence s : d.sentences())
				{
					for (String w : s.words())
					{
						if (!relatedWordsMap.containsKey(w.toLowerCase()))
						{
							relatedWordsMap.put(w.toLowerCase(), getWordsRelatedTo(w.toLowerCase()));
						}
					}
				}
			}

			System.out.print("Creating file...");
			try
			{
				PrintWriter printWriter = new PrintWriter(file);
				printWriter.print(gson.toJson(relatedWordsMap));
				printWriter.flush();
				printWriter.close();
			} catch (FileNotFoundException e)
			{
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
					switch (f) {
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
	}

	private static String getPos(String sentence, int index)
	{
		ArrayList<Sentence> sentences = (ArrayList<Sentence>) new Document(sentence).sentences();
		int curr_index = index;
		for (Sentence s : sentences)
		{
			if (curr_index >= s.length())
			{
				curr_index -= s.length();
			} else
			{
				return s.posTag(curr_index);
			}
		}

		// TODO: Anything else that needs to be here
		throw new RuntimeException("I don't understand what's happening here");
	}

		private static int[] getOpinionPosition(String sentence, String opinion)
		{
			int curr = 0;

			String[] sentence_array = sentence.split("\\s+");
			String[] opinion_array = opinion.split("\\s+");

			while (curr < sentence_array.length)
			{
				if (sentence_array[curr].equals(opinion_array[0]))
				{
					for (int i = 1; i < opinion_array.length; i++)
					{
						if (!sentence_array[curr + i].equals(opinion_array[i]))
						{
							break;
						}
					}
					return new int[]
							{ curr, curr + opinion_array.length };
				}
				curr++;
			}
			return new int[]
					{ 0, 0 };
		}

		private static ArrayList<Pair<String, String>> processSentence(String sentence)
		{
			ArrayList<Pair<String, String>> wordPos = new ArrayList<Pair<String, String>>();

			ArrayList<Sentence> sentences = (ArrayList<Sentence>) new Document(sentence).sentences();

			for (Sentence s: sentences)
			{
				for (int i = 0; i < s.length(); i ++)
				{
					wordPos.add(new Pair<String, String>(s.word(i), s.posTag(i)));
				}
			}
			return wordPos;
		}

		private static String getText(String sentence, int index)
		{
			ArrayList<Sentence> sentences = (ArrayList<Sentence>) new Document(sentence).sentences();
			int curr_index = index;
			for (Sentence s : sentences)
			{
				if (curr_index >= s.length())
				{
					curr_index -= s.length();
				} else
				{
					return s.word(curr_index);
				}
			}
			return "";
		}


	public static Set<String> getAllWords() {
		return allWords;
	}

	private static int getSize(String sentence)
	{
		Document document = new Document(sentence);
		int size = 0;

		for (Sentence s : document.sentences())
		{
			size += s.length();
		}

		return size;
	}

	public static Set<String> getAllPos() {
		return allPos;
	}
}
