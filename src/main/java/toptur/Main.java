package toptur;

import java.io.File;
import java.util.ArrayList;

/**
 * Main class for the project.
 *
 * Here's the usage:
 *
 *     args[0] -- what task we are going to do
 *     args[1] -- whatever input is needed
 *
 * so
 */
public class Main {

    public static final String DEV_DOCS = "dataset/dev";
    public static final String TEST_DOCS = "dataset/test";
    public static final String ORIGINAL_DOCS = "dataset/original_dataset/docs";

    public static void main(String[] args) {
        if (args.length == 0)
            System.exit(0);

        setup();

        String task = args[0].toLowerCase();
        if (task.equals("train")) {
            ArrayList<Document> devDocs = getAllDocsFrom(DEV_DOCS);

        } else if (task.equals("test")) {
            ArrayList<Document> testDocs = getAllDocsFrom(TEST_DOCS);

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

    private static ArrayList<Document> getAllDocsFrom(String path) {
        ArrayList<Document> docs = new ArrayList<Document>();

        File folder = new File(path);

        if (!folder.exists()) {
            System.err.println("Could not find folder \"" + path + "\"!");
            System.exit(1);
        }

        for(File f : folder.listFiles()) {
            docs.add(Document.fromJson(f));
        }

        return docs;
    }

    private static void setup() {

    }
}
