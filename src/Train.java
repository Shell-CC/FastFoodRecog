import org.opencv.core.Core;

import java.io.File;


public class Train {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    static public void main(String[] args) {
        System.out.println("Hello OpenCV " + Core.VERSION);

        File folder = new File("./res/Arbys/");
        Dictionary dictionary = Dictionary.build(5, folder);
        dictionary.saveCenters();
        dictionary.saveToDatabase();
    }
}
