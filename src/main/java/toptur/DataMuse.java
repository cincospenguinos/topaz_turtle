package toptur;

import com.google.gson.Gson;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;

/**
 * Class that allows access to DataMuse API.
 */
public class DataMuse {

    public static final String API_ENDPOINT = "https://api.datamuse.com/words?";

    /**
     * Returns top three related words of DataMuseWord objects related to the word provided.
     *
     * @param word - String to search
     * @return Array
     */
    public static DataMuseWord[] getWordsRelatedTo(String word) {
        String jsonString = "";
        try {
            URL url = new URL(API_ENDPOINT + "ml=" + word + "&max=3");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            BufferedReader rd = new BufferedReader(new InputStreamReader(conn.getInputStream()));

            String line;
            StringBuilder result = new StringBuilder();
            while ((line = rd.readLine()) != null)
                result.append(line);

            rd.close();

            jsonString = result.toString();
        } catch (MalformedURLException e) {
            e.printStackTrace();
            return null;
        } catch (ProtocolException e) {
            e.printStackTrace();
            return null;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        Gson gson = new Gson();
        return gson.fromJson(jsonString, DataMuseWord[].class);
    }
}