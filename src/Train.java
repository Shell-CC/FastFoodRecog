import org.opencv.core.Core;

import java.io.File;


public class Train {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static final String outDataPath = "./out/data/";
    private static final String inDataPath = "/Users/Shawn/Google Drive/ECEN642/642 Final Proj/Datasets/data/";

    static public void main(String[] args) {
        System.out.println("Hello OpenCV " + Core.VERSION);

        File folder = new File(inDataPath + "Arbys/");
        Dictionary dictionary = Dictionary.build(100, folder);
        dictionary.saveCenters(outDataPath);
        dictionary.saveToDatabase(outDataPath);
    }
}
