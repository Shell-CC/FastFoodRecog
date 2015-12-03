import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class Train {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static final String outDataPath = "./out/data/";
    private static final String inDataPath = "/Users/Shawn/Google Drive/ECEN642/642 Final Proj/Datasets/new_data/";

    static public void main(String[] args) throws IOException{
        System.out.println("Hello OpenCV " + Core.VERSION);

        List<Mat> traindataList = new ArrayList<Mat>();
        List<Integer> trainLabelList = new ArrayList<Integer>();
        readTrainData(outDataPath + "Arbys100Datas.csv", traindataList, trainLabelList);

        Classifier classifier = new Classifier();
        classifier.train(traindataList, trainLabelList);
        classifier.test(traindataList, trainLabelList);
    }

    public static void getTrainData() {
        // Build SURF dictionary

        File folder = new File(inDataPath + "Arbys/");
        Dictionary dictionary = Dictionary.build(100, folder);
        dictionary.saveCenters(outDataPath);
        dictionary.saveToDatabase(outDataPath);
    }

    public static void readTrainData(String filename, List<Mat> trainDataList, List<Integer> trainLabelList) throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        try {
            String line = null;
            String title = reader.readLine();
            while ((line = reader.readLine()) != null) {
                String data[] = line.split(",");
                // get each train data
                BagOfSurf bagOfSurf = new BagOfSurf(Integer.parseInt(data[1]), data[2]);
                trainDataList.add(bagOfSurf.toMat());
                // get each train label
                trainLabelList.add(Integer.parseInt(data[3]));
            }
        } finally {
            reader.close();
        }
    }
}
