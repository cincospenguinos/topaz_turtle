package toptur;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.simple.Document;
import edu.stanford.nlp.simple.Sentence;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Type;
import java.util.*;

/**
 * Main class for the project.
 *
 * Here's the usage:
 *
 * args[0] -- what task we are going to do args[1+] -- whatever input is needed
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

	private static final String PHI_WORD = "_PHI_WORD_";
	private static final String OMEGA_WORD = "_OMEGA_WORD_";

	private static SentiWordNetDictionary sentiWordNetDictionary;
	private static Map<String, String[]> relatedWordsMap;

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

			// TODO: Train opinionated sentence detection learner
			Set<LearnerFeature> sentenceFeatures = new TreeSet<LearnerFeature>();
			sentenceFeatures.add(LearnerFeature.CONTAINS_UNIGRAM);
			sentenceFeatures.add(LearnerFeature.CONTAINS_BIGRAM);

			List<LearnerExample<Sentence, Boolean>> opinionatedSentenceExamples = getOpinionatedSentenceExamples(devArticles);

			System.out.println("Training opinionated sentence tree...");
			long start = System.currentTimeMillis();
			DecisionTree<Sentence, Boolean> opinionatedSentenceTree = new DecisionTree<Sentence, Boolean>(opinionatedSentenceExamples,
					LearnerFeatureManager.getInstance().getIdsFor(sentenceFeatures), 1);
			long end = System.currentTimeMillis();

			System.out.println("Took " + (((double) end - (double)start)) / 1000.0  + " seconds");

			// TODO: Train all the other learners

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
				LearnerFeature f = LearnerFeatureManager.getInstance().getLearnerFeatureFor(featureId);
				Object val = LearnerFeatureManager.getInstance().getValueFor(featureId);

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

	////////////////////////
	// FEATURE PROCESSING //
	////////////////////////

	///////////////////////
	// TRAIN CLASSIFIERS //
	///////////////////////

	//////////////////////
	// TEST CLASSIFIERS //
	//////////////////////

	//////////////////////////////////////
	// EXTRACT/EVALUATE WITH CLASSIFIER //
	//////////////////////////////////////

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
		LearnerFeatureManager manager = LearnerFeatureManager.getInstance();

		for (NewsArticle a : devArticles) {
			Document document = new Document(a.getFullText());

			for (Sentence s : document.sentences()) {
				for (LearnerFeature f : LearnerFeature.values()) {
					switch(f) {
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
					}
				}
			}
		}

		// TODO: Assign IDs for all of the other classifier types
	}
}
