import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.io.File;
import java.util.List;


public class Train {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    static public void main(String[] args) {
        System.out.println("Hello OpenCV " + Core.VERSION);

        File folder = new File("./res/Arbys/");
        Dictionary dictionary = Dictionary.build(100, folder);
        dictionary.getCenters();
        List<Mat> labelList = dictionary.getLabelsList();
        for (Mat labels : labelList) {
            System.out.println(labels.size());
        }
    }
}
