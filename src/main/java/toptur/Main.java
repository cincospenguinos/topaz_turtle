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

	public static final String SENTENCES_TRAINING_FILE = TOPTUR_DATA_FOLDER + "sentences_train.vector";
	public static final String SENTENCES_MODEL_FILE = TOPTUR_DATA_FOLDER + "liblinear_models/sentences.model";
	public static final String SENTENCES_TEST_FILE = TOPTUR_DATA_FOLDER + "sentences_test.vector";

	public static final String AGENT_TRAINING_FILE = TOPTUR_DATA_FOLDER + "agent_train.vector";
	public static final String AGENT_MODEL_FILE = TOPTUR_DATA_FOLDER + "liblinear_models/agent.model";
	public static final String AGENT_TEST_FILE = TOPTUR_DATA_FOLDER + "agent_test.vector";

	public static final String TARGET_TRAINING_FILE = TOPTUR_DATA_FOLDER + "target_train.vector";
	public static final String TARGET_MODEL_FILE = TOPTUR_DATA_FOLDER + "liblinear_models/target.model";
	public static final String TARGET_TEST_FILE = TOPTUR_DATA_FOLDER + "target_test.vector";

	public static final String OPINION_TRAINING_FILE = TOPTUR_DATA_FOLDER + "opinion_train.vector";
	public static final String OPINION_TEST_FILE = TOPTUR_DATA_FOLDER + "opinion_test.vector";
	public static final String OPINION_MODEL_FILE = TOPTUR_DATA_FOLDER + "liblinear_models/opinion.model";

	public static final String SENTIMENT_TRAINING_FILE = TOPTUR_DATA_FOLDER + "sentiment_train.vector";
	public static final String SENTIMENT_TEST_FILE = TOPTUR_DATA_FOLDER + "sentiment_test.vector";
	public static final String SENTIMENT_MODEL_FILE = TOPTUR_DATA_FOLDER + "liblinear_models/sentiment.model";

	public static final String LIB_LINEAR_FEATURE_MANAGER_FILE = TOPTUR_DATA_FOLDER
			+ "/lib_linear_feature_manager.json";
	public static final String SENTI_WORD_NET_FILE = "sentiwordnet.txt";
	public static final String RELATED_WORDS_FILE = TOPTUR_DATA_FOLDER + "related_words.json";

	private static SentiWordNetDictionary sentiWordNetDictionary;
	private static Map<String, String[]> relatedWordsMap;

	private static final String PHI_WORD = "__PHI__";
	private static final String PHI_POS = "__PHI_POS__";
	private static final String OMEGA_WORD = "__OMEGA__";
	private static final String OMEGA_POS = "__OMEGA_POS__";
	
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




	//////////////////////
	// MAIN //
	//////////////////////

	public static void main(String[] args)
	{
		if (args.length == 0)
			System.exit(0);

		System.out.println("Gathering SentiWordNet dictionary...");
		getSentiWordNet();
		getRelatedWordsMap();

		String task = args[0].toLowerCase();

		if (task.equals("train"))
		{
			ArrayList<NewsArticle> devArticles = getAllDocsFrom(DEV_DOCS);

			// Train to detect opinionated sentences
			createSentencesVectorFile(devArticles, SENTENCES_TRAINING_FILE);
			trainLibLinear(SENTENCES_TRAINING_FILE, SENTENCES_MODEL_FILE);

			// Train to detect opinion agents
			createTrainVectorFileAgent(devArticles, AGENT_TRAINING_FILE);
			trainLibLinear(AGENT_TRAINING_FILE, AGENT_MODEL_FILE);

			// Train to detect opinion agents
			createTrainVectorFileTarget(devArticles, TARGET_TRAINING_FILE);
			trainLibLinear(TARGET_TRAINING_FILE, TARGET_MODEL_FILE);

			// Train to detect opinions in the sentence
			createOpinionVectorFile(devArticles, OPINION_TRAINING_FILE);
			trainLibLinear(OPINION_TRAINING_FILE, OPINION_MODEL_FILE);

            // Train to determine polarity of a given sentence
			createPolarityVectorFile(devArticles, SENTIMENT_TRAINING_FILE);
			trainLibLinear(SENTIMENT_TRAINING_FILE, SENTIMENT_MODEL_FILE);

            LibLinearFeatureManager.saveInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);
            
            System.out.println("Finished Training");

		} else if (task.equals("test"))
		{
			TreeSet<EvaluationOption> evalOptions = new TreeSet<EvaluationOption>();

			if (args.length == 1)
			{
				for (EvaluationOption e : EvaluationOption.values())
					evalOptions.add(e);
			} else
			{
				for (int i = 1; i < args.length; i++)
				{
					EvaluationOption e = EvaluationOption.valueOf(args[i].replaceAll("-", "").toUpperCase());
					evalOptions.add(e);
				}
			}

			ArrayList<NewsArticle> testArticles = getAllDocsFrom(TEST_DOCS);

			createSentencesVectorFile(testArticles, SENTENCES_TEST_FILE);
			createTrainVectorFileAgent(testArticles, AGENT_TEST_FILE);
			createTrainVectorFileTarget(testArticles, TARGET_TEST_FILE);
			createOpinionVectorFile(testArticles, OPINION_TEST_FILE);
			createPolarityVectorFile(testArticles, SENTIMENT_TEST_FILE);

			testLibLinear(SENTENCES_TEST_FILE, SENTENCES_MODEL_FILE, "/dev/null");
			testLibLinear(AGENT_TEST_FILE, AGENT_MODEL_FILE, "/dev/null");
			testLibLinear(TARGET_TEST_FILE, TARGET_MODEL_FILE, "/dev/null");
			testLibLinear(OPINION_TEST_FILE, OPINION_MODEL_FILE, "/dev/null");
			testLibLinear(SENTIMENT_TEST_FILE, SENTIMENT_MODEL_FILE, "/dev/null");

			// Extract the opinions
			// Let's time how long it takes
			System.out.println("Starting extraction...");
			long startTime = System.currentTimeMillis();
			for (NewsArticle a : testArticles)
				extractOpinionFramesForArticle(a);
			long endTime = System.currentTimeMillis();

			System.out.println(((double) endTime - startTime) / 1000.0 + " seconds");

			if (evalOptions.size() == 1 && evalOptions.contains(EvaluationOption.TARGET))
				evaluateTargetClassifier(testArticles);
			else if(evalOptions.size() == 1 && evalOptions.contains(EvaluationOption.AGENT))
				evaluateAgentClassifier(testArticles);
			else
				evaluateExtractedOpinions(testArticles, evalOptions);

		} else if (task.equals("extract"))
		{
			for (int i = 1; i < args.length; i++)
			{
				File file = new File(args[i]);

				if (file.exists())
				{
					NewsArticle article = new NewsArticle(file);
					extractOpinionFramesForArticle(article);
					System.out.println(article);
				} else
				{
					System.err.println("\"" + args[i] + "\" could not be found!");
				}
			}

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

	/**
	 * Creates a vector file for the polarity of a given sentence.
	 *
	 * @param articles - articles to create file from
	 * @param vectorFileName - name of the file
	 */
	private static void createPolarityVectorFile(List<NewsArticle> articles, String vectorFileName) {
		StringBuilder vectorFileBuilder = new StringBuilder();

		for (NewsArticle a : articles) {
			for (Opinion o : a.getGoldStandardOpinions().values()) {
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
	private static void createSentencesVectorFile(ArrayList<NewsArticle> articles, String nameOfVectorFile) {
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

				String line = generateSentenceLineVectorFileString(null, s);
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
	 * Creates a single line for LibLinear of the given sentence from the news article provided.
	 *
	 * @param article -
	 * @param sentence -
	 * @return String for liblinear to mange
	 */
	private static String generateSentenceLineVectorFileString(NewsArticle article, Sentence sentence) {
		StringBuilder vectorLineBuilder = new StringBuilder();
		TreeMap<Integer, Object> libLinearFeatureVector = new TreeMap<Integer, Object>();
		LibLinearFeatureManager manager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

		// The label for this sentence
		if (article != null) {
			if (article.sentenceHasOpinion(sentence.toString()))
				vectorLineBuilder.append(1);
			else
				vectorLineBuilder.append(0);
		}

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

				case HAS_WORD_RELATED_TO_OTHER_WORD:
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
	
	
	private static void createTrainVectorFileAgent(ArrayList<NewsArticle> articles, String nameOfVectorFile)
	{
		StringBuilder vectorFileBuilder = new StringBuilder();
		LibLinearFeatureManager manager = LibLinearFeatureManager
				.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);
		
		for(NewsArticle article: articles)
		{
			for(String opinion_phrase: article.getGoldStandardOpinions().keySet())
			{
				Opinion opinion_record = article.getGoldStandardOpinions().get(opinion_phrase);
				
				int[] position = getOpinionPosition(opinion_record.sentence, opinion_phrase);
				
				vectorFileBuilder.append(createVectorFileAgent(article, opinion_phrase, position[0], position[1]));

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
	
	private static void createSingleOpinionAgentTarget(String sentence, ArrayList<String> words, int index, String nameOfVectorFile)
	{
		LibLinearFeatureManager manager = LibLinearFeatureManager
				.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);
		
		String word;
		String pos;

		if (words.get(index).equals(W_WORD))
		{
			word = W_WORD;
			pos = W_POS;
		}
		else if (words.get(index).equals(NULL_WORD))
		{
			word = NULL_WORD;
			pos = NULL_POS;
		}
		else
		{
			word = getText(sentence, index);
			pos = getPos(sentence, index);
		}
		
		StringBuilder vectorLineBuilder = new StringBuilder();
		vectorLineBuilder.append(0);
	    vectorLineBuilder.append(' ');
		TreeMap<Integer, String> featureVector = new TreeMap<Integer, String>();

		// Creating the feature vectors
		for (LibLinearFeatureManager.LibLinearFeature feature : LibLinearFeatureManager.LibLinearFeature
				.values())
		{
			int id;		
			
			switch (feature)
			{
			case PREVIOUS_UNIGRAM:
				String previous;

				if (word.equals(W_WORD))
					previous = W_PREV;
				else if (word.equals(NULL_WORD))
					previous = NULL_PREV;
				else if (index > 0)
					previous = getText(sentence, index - 1);
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
				else if (index < getSize(sentence) - 1)
					next = getText(sentence, index + 1);
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
				else if (index > 0)
					previousPos = getPos(sentence, index - 1);
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
				else if (index < getSize(sentence) - 1)
					nextPos = getPos(sentence, index + 1);
				else
					nextPos = OMEGA_POS;

				id = manager.getIdFor(feature, nextPos);
				featureVector.put(id, id + ":1");
				break;
			case OBJECTIVITY_OF_WORD:
				id = manager.getIdFor(feature, "");
				int objectivity = sentiWordNetDictionary.getObjectivityOf(word);
				featureVector.put(id, id + ":" + objectivity);
				break;
			}	
			
		}

		for (String s : featureVector.values())
		{
			vectorLineBuilder.append(s);
			vectorLineBuilder.append(' ');
		}

		try
		{
			PrintWriter vectorFile = new PrintWriter(nameOfVectorFile);
			vectorFile.print(vectorLineBuilder.toString());
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
	 * @param article
	 * @param opinion_phrase
	 * @param start
	 * @param end
	 */
	private static String createVectorFileAgent(NewsArticle article, String opinion_phrase, int start, int end)
	{
		StringBuilder vectorFileBuilder = new StringBuilder();
		LibLinearFeatureManager manager = LibLinearFeatureManager
				.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

		HashMap<String, Opinion> gold = article.getGoldStandardOpinions();
		
		Opinion opinion = gold.get(opinion_phrase);
		
		String sentence = opinion.sentence;
		ArrayList<String> words = new ArrayList<String>();
		
		String agent;
		
		if (opinion.agent == null)
		{
			agent = "null";
		}
		else
		{
			agent = opinion.agent;
		}
		
		for (int i = start; i < end; i++)
		{
			words.add(getText(sentence, i));
		}
		
		words.add(W_WORD);
		words.add(NULL_WORD);
		
		for (int i = 0; i < words.size(); i++)
		{
			String word;
			String pos;

			if (words.get(i).equals(W_WORD))
			{
				word = W_WORD;
				pos = W_POS;
			}
			else if (words.get(i).equals(NULL_WORD))
			{
				word = NULL_WORD;
				pos = NULL_POS;
			}
			else
			{
				word = words.get(i);
				pos = getPos(sentence, start+i);
			}
						
			StringBuilder vectorLineBuilder = new StringBuilder();
			TreeMap<Integer, Object> featureVector = new TreeMap<Integer, Object>();
			
			if (word.equals(W_WORD) && agent.equals("w"))
			{
				vectorLineBuilder.append(1);
			}
			else if (word.equals(NULL_WORD) && agent.equals("null"))
			{
				vectorLineBuilder.append(1);
			}
			else if (word.equalsIgnoreCase(agent))
			{
				vectorLineBuilder.append(1);
			} 
			else
			{
				vectorLineBuilder.append(0);
			}

			// Creating the feature vectors
			for (LibLinearFeatureManager.LibLinearFeature feature : LibLinearFeatureManager.LibLinearFeature
					.values())
			{
				int id;		
				
				switch (feature)
				{
				case PREVIOUS_UNIGRAM:
					String previous;

					if (word.equals(W_WORD))
						previous = W_PREV;
					else if (word.equals(NULL_WORD))
						previous = NULL_PREV;
					else if (i > 0)
						previous = getText(sentence, start + i - 1);
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
					else if (i < getSize(sentence) - 1)
						next = getText(sentence, start + i + 1);
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
					else if (i > 0)
						previousPos = getPos(sentence, start + i - 1);
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
					else if (i < getSize(sentence) - 1)
						nextPos = getPos(sentence, start + i + 1);
					else
						nextPos = OMEGA_POS;

					id = manager.getIdFor(feature, nextPos);
					featureVector.put(id, id + ":1");
					break;
				case OBJECTIVITY_OF_WORD:
					id = manager.getIdFor(feature, "");
					int objectivity = sentiWordNetDictionary.getObjectivityOf(word);
					featureVector.put(id, id + ":" + objectivity);
					break;
				}	
				
			}

			for (Map.Entry<Integer, Object> e : featureVector.entrySet())
			{
				vectorLineBuilder.append(" ");
				vectorLineBuilder.append(e.getKey());
				vectorLineBuilder.append(":");

				if (e.getValue() instanceof Boolean)
					vectorLineBuilder.append(1);
				else
					vectorLineBuilder.append(0);
			}

			vectorFileBuilder.append(vectorLineBuilder.toString());
			vectorFileBuilder.append("\n");
		}
	
		return vectorFileBuilder.toString();
		
	}
	
	private static void createTrainVectorFileTarget(ArrayList<NewsArticle> articles, String nameOfVectorFile)
	{
		StringBuilder vectorFileBuilder = new StringBuilder();
		LibLinearFeatureManager manager = LibLinearFeatureManager
				.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);
		
		for(NewsArticle article: articles)
		{
			for(String opinion_phrase: article.getGoldStandardOpinions().keySet())
			{
				Opinion opinion_record = article.getGoldStandardOpinions().get(opinion_phrase);
				
				int[] position = getOpinionPosition(opinion_record.sentence, opinion_phrase);
				
				vectorFileBuilder.append(createVectorFileTarget(article, opinion_phrase, position[0], position[1]));

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
	 * @param article
	 * @param opinion_phrase
	 * @param start
	 * @param end
	 */
	private static String createVectorFileTarget(NewsArticle article, String opinion_phrase, int start, int end)
	{
		StringBuilder vectorFileBuilder = new StringBuilder();
		LibLinearFeatureManager manager = LibLinearFeatureManager
				.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

		HashMap<String, Opinion> gold = article.getGoldStandardOpinions();

		Opinion opinion = gold.get(opinion_phrase);

		String sentence = opinion.sentence;
		ArrayList<String> words = new ArrayList<String>();
		
		String target;
		
		if (opinion.target == null)
		{
			target = "null";
		}
		else
		{
			target = opinion.target;
		}

		for (int i = start; i < end; i++)
		{
			words.add(getText(sentence, i));
		}

		words.add(W_WORD);
		words.add(NULL_WORD);

		for (int i = 0; i < words.size(); i++)
		{
			String word;
			String pos;

			if (words.get(i).equals(W_WORD))
			{
				word = W_WORD;
				pos = W_POS;
			}
			else if (words.get(i).equals(NULL_WORD))
			{
				word = NULL_WORD;
				pos = NULL_POS;
			}
			else
			{
				word = words.get(i);
				pos = getPos(sentence, start+i);
			}

			StringBuilder vectorLineBuilder = new StringBuilder();
			TreeMap<Integer, Object> featureVector = new TreeMap<Integer, Object>();
			
			if (word.equals(W_WORD) && target.equals("w"))
			{
				vectorLineBuilder.append(1);
			}
			else if (word.equals(NULL_WORD) && target.equals("null"))
			{
				vectorLineBuilder.append(1);
			}
			else if (word.equalsIgnoreCase(target))
			{
				vectorLineBuilder.append(1);
			} else
			{
				vectorLineBuilder.append(0);
			}

			// Creating the feature vectors
			for (LibLinearFeatureManager.LibLinearFeature feature : LibLinearFeatureManager.LibLinearFeature
					.values())
			{
				int id;

				switch (feature)
				{
				case PREVIOUS_UNIGRAM:
					String previous;

					if (word.equals(W_WORD))
						previous = W_PREV;
					else if (word.equals(NULL_WORD))
						previous = NULL_PREV;
					else if (i > 0)
						previous = getText(sentence, start + i - 1);
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
					else if (i < getSize(sentence) - 1)
						next = getText(sentence, start + i + 1);
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
					else if (i > 0)
						previousPos = getPos(sentence, start + i - 1);
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
					else if (i < getSize(sentence) - 1)
						nextPos = getPos(sentence, start + i + 1);
					else
						nextPos = OMEGA_POS;

					id = manager.getIdFor(feature, nextPos);
					featureVector.put(id, id + ":1");
					break;
				case OBJECTIVITY_OF_WORD:
					id = manager.getIdFor(feature, "");
					int objectivity = sentiWordNetDictionary.getObjectivityOf(word);
					featureVector.put(id, id + ":" + objectivity);
					break;
				}

			}

			for (Map.Entry<Integer, Object> e : featureVector.entrySet())
			{
				vectorLineBuilder.append(" ");
				vectorLineBuilder.append(e.getKey());
				vectorLineBuilder.append(":");

				if (e.getValue() instanceof Boolean)
					vectorLineBuilder.append(1);
				else
					vectorLineBuilder.append(0);
			}

			vectorFileBuilder.append(vectorLineBuilder.toString());
			vectorFileBuilder.append("\n");
		}

		return vectorFileBuilder.toString();
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
		LibLinearFeatureManager libLinearFeatureManager = LibLinearFeatureManager
				.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

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

					int id = libLinearFeatureManager.getIdFor(feature, bigram);
					libLinearFeatureVector.put(id, true);
				}
				break;
			case OBJECTIVITY_OF_SENTENCE:
				int id = libLinearFeatureManager.getIdFor(feature, "");
				int objectivity = 0;

				for (String w : words) {
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
		for (LibLinearFeatureManager.LibLinearFeature feature : LibLinearFeatureManager.LibLinearFeature.values())
		{
			int id;
			switch (feature)
			{
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
					int objectivity = sentiWordNetDictionary.getObjectivityOf(thisWord);
					stupidMap.put(id, id + ":" + objectivity);
					break;
			}
		}

		for (String s : stupidMap.values())
		{
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
	private static void createOpinionVectorFile(ArrayList<NewsArticle> articles, String opinionTrainingFile)
	{
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
			PrintWriter vectorFile = new PrintWriter(opinionTrainingFile);
			vectorFile.print(vectorFileBuilder.toString().trim());
			vectorFile.flush();
			vectorFile.close();
		} catch (FileNotFoundException e)
		{
			e.printStackTrace();
		}
	}

	/**
	 * Extracts polarity of a given sentence
	 *
	 * @param sentence -
	 * @return String
	 */
	private static String extractPolarityOfSentence(Sentence sentence) {
		StringBuilder vectorFileBuilder = new StringBuilder();
		LibLinearFeatureManager manager = LibLinearFeatureManager.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);

		String vectorFileName = "polarity_herp.vector";

		vectorFileBuilder.append(0);
		vectorFileBuilder.append(' ');

		String line = generateSentenceLineVectorFileString(null, sentence);
		vectorFileBuilder.append(line);
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

		try {
			Runtime.getRuntime().exec("./liblinear_predict " + vectorFileName + " " + SENTIMENT_MODEL_FILE + " herp.txt");
			Thread.sleep(100); // to give the machine time to print to file
			Scanner s = new Scanner(new File("herp.txt"));
			int id = Integer.parseInt(s.next());
			s.close();

			return Opinion.fromSentimentId(id);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		return null;
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
		} catch (InterruptedException e)
		{
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
			Thread.sleep(500); // To give LibLinear enough time to output to file
			Scanner derp = new Scanner(Runtime.getRuntime().exec("cat output.txt").getInputStream());
			int i = 0;
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
	 * Extracts all opinion frames found in a given news article.
	 *
	 * @param article - Article to extract from
	 */
	private static void extractOpinionFramesForArticle(NewsArticle article)
	{
		Document doc = new Document(article.getFullText());

		for (Sentence s : doc.sentences())
		{
			if (sentenceContainsOpinion(s))
			{
				for (String opinionExpression : getAllOpinionsInSentence(s))
				{
					Opinion o = new Opinion();
					o.opinion = opinionExpression;
					o.sentence = s.toString();
					o.sentiment = extractPolarityOfSentence(s);
					o.agent = extractAgentFrom(o);
					o.target = extractTargetFrom(o);

					article.addExtractedOpinion(o);
				}
			}
		}
	}

	/**
	 * Extracts all opinions from the sentence provided.
	 *
	 * @param s
	 *            - Sentence to get opinions from
	 * @return Set of type String
	 */
	private static Set<String> getAllOpinionsInSentence(Sentence s)
	{
		String name = "some_file.vector";

		int previousTag = 0;

		TreeSet<String> allOpinions = new TreeSet<String>();

		StringBuilder opinionBuilder = new StringBuilder();
		for (int i = 0; i < s.words().size(); i++)
		{
			String previousWord, nextWord, previousPos, nextPos;

			if (i > 0)
			{
				previousWord = s.word(i - 1);
				previousPos = s.posTag(i - 1);
			} else
			{
				previousWord = PHI_WORD;
				previousPos = PHI_POS;
			}

			if (i < s.words().size() - 1)
			{
				nextWord = s.word(i + 1);
				nextPos = s.posTag(i + 1);
			} else
			{
				nextWord = OMEGA_WORD;
				nextPos = OMEGA_POS;
			}

			String word = s.word(i);
			String pos = s.posTag(i);
			createSingleWordVectorFile(previousWord, word, nextWord, previousPos, pos, nextPos, name);

			try
			{
				Runtime.getRuntime().exec("./liblinear_predict " + name + " " + SENTENCES_MODEL_FILE + " output.txt");
				Thread.sleep(100); // To give LibLinear enough time to output to file

				Scanner scanner = new Scanner(new File("output.txt"));

				while (scanner.hasNextLine())
				{
					int label = Integer.parseInt(scanner.nextLine());

					if (label != 0)
					{
						opinionBuilder.append(word);
						opinionBuilder.append(" ");
					} else if (opinionBuilder.length() > 0)
					{
						allOpinions.add(opinionBuilder.toString());
						opinionBuilder = new StringBuilder();
					}
				}
			} catch (IOException e)
			{
				e.printStackTrace();
			} catch (InterruptedException e)
			{
				e.printStackTrace();
			}
		}

		return allOpinions;
	}
	
	private static String extractAgentFrom(Opinion o)
	{
		String sentence = o.sentence;
		int[] pos = getOpinionPosition(o.sentence, o.opinion);
		
		ArrayList<String> words = new ArrayList<String>();
		
		for (int i = pos[0]; i < pos[1]; i++)
		{
			words.add(getText(sentence, i));
		}
		
		words.add(W_WORD);
		words.add(NULL_WORD);
		
		HashMap<String, Double> confidences = new HashMap<String, Double>();

		for (int i = 0; i < words.size(); i++) 
		{
			String name = "some_file_" + i + ".vector";
			createSingleOpinionAgentTarget(sentence, words, i, name);
			
			try
			{
				Runtime.getRuntime().exec("./liblinear_predict " + name + " " + AGENT_MODEL_FILE + " output.txt");
				Thread.sleep(100); // To give LibLinear enough time to output to file

				Scanner scanner = new Scanner(new File("output.txt"));

				while (scanner.hasNextLine())
				{
					double label = Double.parseDouble(scanner.nextLine());

					confidences.put(words.get(i), label);
				}
			} catch (IOException e)
			{
				e.printStackTrace();
			} catch (InterruptedException e)
			{
				e.printStackTrace();
			}
		}
		
		double max = confidences.get(W_WORD);
		String max_word = W_WORD;
		for (String word: words)
		{
			if (confidences.get(word) > max)
			{
				max = confidences.get(word);
				max_word = word;
			}
		}
		
		if (max_word.equals(W_WORD))
			return "w";
		else if (max_word.equals(NULL_WORD))
			return "null";
		else
			return max_word;
	}
	
	private static String extractTargetFrom(Opinion o)
	{
		String sentence = o.sentence;
		int[] pos = getOpinionPosition(o.sentence, o.opinion);
		
		LibLinearFeatureManager manager = LibLinearFeatureManager
				.getInstance(LIB_LINEAR_FEATURE_MANAGER_FILE);
		
		ArrayList<String> words = new ArrayList<String>();
		
		for (int i = pos[0]; i < pos[1]; i++)
		{
			words.add(getText(sentence, i));
		}
		
		words.add(W_WORD);
		words.add(NULL_WORD);
		
		HashMap<String, Double> confidences = new HashMap<String, Double>();

		for (int i = 0; i < words.size(); i++) 
		{
			String name = "some_file_" + i + ".vector";
			createSingleOpinionAgentTarget(sentence, words, i, name);
			
			try
			{
				Runtime.getRuntime().exec("./liblinear_predict " + name + " " + TARGET_MODEL_FILE + " output.txt");
				Thread.sleep(100); // To give LibLinear enough time to output to file

				Scanner scanner = new Scanner(new File("output.txt"));

				while (scanner.hasNextLine())
				{
					double label = Double.parseDouble(scanner.nextLine());

					confidences.put(words.get(i), label);
				}
			} catch (IOException e)
			{
				e.printStackTrace();
			} catch (InterruptedException e)
			{
				e.printStackTrace();
			}
		}
		
		double max = confidences.get(W_WORD);
		String max_word = W_WORD;
		for (String word: words)
		{
			if (confidences.get(word) > max)
			{
				max = confidences.get(word);
				max_word = word;
			}
		}
		
		if (max_word.equals(W_WORD))
			return "w";
		else if (max_word.equals(NULL_WORD))
			return "null";
		else
			return max_word;
	}

	/**
	 * Evaluates the extracted opinions, given whatever evaluation options desired.
	 *
	 * @param articles
	 * @param evaluationOptions
	 */
	private static void evaluateExtractedOpinions(ArrayList<NewsArticle> articles,
			Set<EvaluationOption> evaluationOptions)
	{
		double totalFScore = 0.0;

		System.out.println("Name\tPrecision\tRecall\tFScore");

		for (NewsArticle article : articles)
		{
			HashMap<String, Opinion> extractedOpinions = (HashMap<String, Opinion>) article.getExtractedOpinions()
					.clone();
			HashMap<String, Opinion> goldStandardOpinions = (HashMap<String, Opinion>) article.getGoldStandardOpinions()
					.clone();

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

			double precision = truePositives / (extractedOpinions.size() + correct.size());
			double recall = truePositives / (goldStandardOpinions.size() + correct.size());
			double fscore = 2 * ((precision * recall) / (Math.max(1, precision + recall)));

			System.out.println(article.getDocumentName() + "\t" + precision + "\t" + recall + "\t" + fscore);
			totalFScore += fscore;
		}

		System.out.println("\nTOTAL FSCORE: " + totalFScore / articles.size());
	}

	/**
	 * Returns precision, recall, and Fscore in that order.
	 *
	 * @param articles
	 */
	private static void evaluateAgentClassifier(List<NewsArticle> articles)
	{
		double totalFScore = 0.0;

		System.out.println("Name\tPrecision\tRecall\tFScore");

		double correctLabel = 0.0;
		double nonNullLabels = 0.0;
		double nonNullCorrect = 0.0;

		for (NewsArticle article : articles) {
			HashMap<String, Opinion> goldStandardOpinions = (HashMap<String, Opinion>) article.getGoldStandardOpinions()
					.clone();
			for (Opinion goldStandard : goldStandardOpinions.values())
			{
				String classifier_result = extractAgentFrom(goldStandard);

				if (!goldStandard.agent.equals("null"))
				{
					nonNullLabels += 1.0;
					if (classifier_result.equals(goldStandard.agent))
					{
						correctLabel += 1.0;
						nonNullCorrect += 1.0;
					}
				}
				else if (classifier_result.equals(goldStandard.agent))
				{
					correctLabel += 1.0;
				}
			}

			double precision = correctLabel / goldStandardOpinions.size();
			double recall = nonNullCorrect / nonNullLabels;
			double fscore = 2 * ((precision * recall) / precision + recall);

			System.out.println(article.getDocumentName() + "\t" + precision + "\t" + recall + "\t" + fscore);

			totalFScore += fscore;
		}

		System.out.println("\nTOTAL FSCORE\t" + totalFScore / articles.size());
	}


	/**
	 * Returns precision, recall, and fscore in that order
	 * @param articles
	 * @return
	 */
	private static void evaluateTargetClassifier(List<NewsArticle> articles)
	{
		double totalFScore = 0.0;

		for (NewsArticle article : articles) {
			HashMap<String, Opinion> goldStandardOpinions = (HashMap<String, Opinion>) article.getGoldStandardOpinions()
					.clone();

			double correctLabel = 0.0;
			double nonNullLabels = 0.0;
			double nonNullCorrect = 0.0;


			for (Opinion goldStandard : goldStandardOpinions.values())
			{
				String classifier_result = extractTargetFrom(goldStandard);

				if (!goldStandard.target.equals("null"))
				{
					nonNullLabels += 1.0;
					if (classifier_result.equals(goldStandard.target))
					{
						correctLabel += 1.0;
						nonNullCorrect += 1.0;
					}
				}
				else if (classifier_result.equals(goldStandard.target))
				{
					correctLabel += 1.0;
				}
			}

			double precision = correctLabel / goldStandardOpinions.size();
			double recall = nonNullCorrect / nonNullLabels;
			double fscore = 2 * ((precision * recall) / precision + recall);

			System.out.println(article.getDocumentName() + "\t" + precision + "\t" + recall + "\t" + fscore);

			totalFScore += fscore;
		}

		System.out.println("\nTOTAL FSCORE\t" + totalFScore / articles.size());
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

	private static String[] getWordsRelatedTo(String word) {
		DataMuseWord[] dataMuseWords = DataMuse.getWordsRelatedTo(word.toLowerCase());
		String[] relatedWords = new String[dataMuseWords.length];

		for (int i = 0; i < relatedWords.length; i++)
			relatedWords[i] = dataMuseWords[i].word;

		return relatedWords;
	}
	
	private static int[] getOpinionPosition(String sentence, String opinion)
	{
		int curr = 0;
		
		String[] sentence_array = sentence.split("\\s+");
		String[] opinion_array = opinion.split("\\s+");
		
		while (curr < sentence_array.length)
		{
			if(sentence_array[curr].equals(opinion_array[0]))
			{
				for (int i = 1; i < opinion_array.length; i++)
				{
					if(!sentence_array[curr+i].equals(opinion_array[i]))
					{
						break;
					}
				}
				return new int[] {curr, curr+opinion_array.length};
			}	
			curr++;
		}
		return new int[] {0,0};
	}
	
	private static String getText(String sentence, int index)
	{
		ArrayList<Sentence> sentences = (ArrayList<Sentence>) new Document(sentence).sentences();
		int curr_index = index;
		for(Sentence s: sentences)
		{
			if (curr_index >= s.length()) 
			{
				curr_index -= s.length();
			}
			else
			{
				return s.word(curr_index);
			}
		}
		return "";
	}
	
	private static String getPos(String sentence, int index)
	{
		ArrayList<Sentence> sentences = (ArrayList<Sentence>) new Document(sentence).sentences();
		int curr_index = index;
		for(Sentence s: sentences)
		{
			if (curr_index >= s.length()) 
			{
				curr_index -= s.length();
			}
			else
			{
				return s.posTag(curr_index);
			}
		}
		return "";
	}
	
	private static int getSize(String sentence)
	{
		Document document = new Document(sentence);
		int size = 0;
		
		for (Sentence s: document.sentences())
		{
			size += s.length();
		}
		
		return size;
	}

}
