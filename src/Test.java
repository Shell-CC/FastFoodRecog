import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.io.FileReader;

/**
 * Created by Shawn on 12/3/15.
 */
public class Test {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }


    static public void main(String[] args) throws Exception {
        FoodImage image = new FoodImage();
        image.read("./testImage/test1.jpg");
        Mat mask = image.extractBackgroundMask();
        Mat surfs = image.extractSurf(mask);
        //image.drawFeatures("./testImage/test1feat.jpg");
        //image.drawBackground("./testImage/test1back.jpg");

        // train svm
        Classifier classifier = new Classifier();
        classifier.load(Train.outDataPath + "svm.xml");

        // extract bag of surf

//        int l = classifier.predict()
    }

}
