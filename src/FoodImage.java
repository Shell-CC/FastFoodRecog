import org.opencv.core.*;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * Created by Shawn on 11/25/15.
 * Food image info.
 */

public class FoodImage {

    private Mat image;

    private Mat backgroundMask;
    private MatOfKeyPoint features;

    /**
     * Empty constructor.
     */
    public FoodImage() {
        this(new Mat());
    }

    /**
     * Construct food image from an OpenCV mat(org.opencv.*).
     * @param image Image in OpenCV MAT format.
     */
    public FoodImage(Mat image) {
        this.image = image;
        this.features = new MatOfKeyPoint();
    }

    /**
     * Check if the image is empty.
     * @return True if the image empty.
     */
    public boolean isEmpty() {
        return image.total() == 0;
    }

    /**
     * Read image from file.
     * @param filename Path name of the file.
     * @throws IOException If file path is wrong or image format is not supported.
     */
    public void read(String filename) throws IOException{
        image = Highgui.imread(filename);
        if (image.total() == 0) {
            throw new FileNotFoundException(filename);
        }
    }


    /**
     * Write the image to the file.
     * @param filename File path to be written.
     * @throws Exception If image format is not supported.
     */
    public void write(String filename) throws Exception{
        if (!Highgui.imwrite(filename, image)) {
            throw new InvalidImageFormatException(filename);
        }
    }

    public Mat foregroundMask() {

        // Get threshold binary image
        Mat binaryImage = new Mat();
        Imgproc.cvtColor(image, binaryImage, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(binaryImage, binaryImage, 128, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
//        Mat foreground = new Mat();
//        Imgproc.morphologyEx(grayImage, foreground, Imgproc.MORPH_OPEN, Mat.ones(50, 50, CvType.CV_8U));

        // Get the distance transform and then normalize and threshold it.
        Mat dist = new Mat();
        Imgproc.distanceTransform(binaryImage, dist, Imgproc.CV_DIST_L2, 3);
//        Core.normalize(dist, dist, 0.0, 1.0, Core.NORM_MINMAX);
        Imgproc.threshold(dist, dist, 128, 255, Imgproc.THRESH_BINARY);

        return binaryImage;
    }

    public Mat extractBackgroundMask() {
        Mat diff = new Mat();
        Imgproc.cvtColor(image, diff, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(diff, diff, 64, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        Imgproc.morphologyEx(diff, diff, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8U), new Point(-1, -1), 10);
//        Mat foreground = new Mat();
//        image.copyTo(foreground, diff);
//        Highgui.imwrite("./res/mask.png", diff);
        this.backgroundMask = diff;
        return diff;
    }


    public Mat extractBackgroundMask(Mat background) {
        if (background.rows() == 0) {
            System.out.println("Error reading background image");
        }
        Mat diff = new Mat();
        Core.absdiff(image, background, diff);
        Imgproc.cvtColor(diff, diff, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(diff, diff, 10, 255, Imgproc.THRESH_BINARY);
        Imgproc.morphologyEx(diff, diff, Imgproc.MORPH_OPEN, Mat.ones(3, 3, CvType.CV_8U), new Point(-1, -1), 10);
//        Highgui.imwrite("./res/trainMask.jpg", diff);
//        Mat foreground = new Mat();
//        image.copyTo(foreground, diff);
//        Highgui.imwrite("./res/trainWithMask.png", foreground);
        this.backgroundMask = diff;
        return diff;
    }


    public Mat extractSurf(Mat mask) {
        Mat descriptor = new Mat();

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.SURF);
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SURF);

        if (mask.empty()) {
            detector.detect(image, features);
        } else {
            detector.detect(image, features, mask);
        }
//        System.out.println("Number of keypoints: " + features.rows());
//        drawFeatures(features);
        extractor.compute(image, features, descriptor);
//        System.out.println("Descriptors in keypoint: " + descriptor.size());

        return descriptor;
    }

    public Mat extractSurf() {
        return extractSurf(new Mat());
    }

    public void drawFeatures(String filename) throws Exception {
        if (features.total() != 0) {
            Mat imageWithFeatures = new Mat();
            Features2d.drawKeypoints(image, features, imageWithFeatures);
            if (!Highgui.imwrite(filename, imageWithFeatures)) {
                throw new InvalidImageFormatException(filename);
            }
        } else {
            throw new EmptyContentException("features");
        }
    }


    public void drawBackground(String filename) throws Exception {
        if (features.total() != 0) {
            Mat imageWithMask = new Mat();
            image.copyTo(imageWithMask, backgroundMask);
            if (!Highgui.imwrite(filename, imageWithMask)) {
                throw new InvalidImageFormatException(filename);
            }
        } else {
            throw new EmptyContentException("background");
        }
    }


    class EmptyContentException extends Exception {
        public EmptyContentException(String message) {
            super(message);
        }
    }
}

class InvalidImageFormatException extends IOException {
    public InvalidImageFormatException(String message) {
        super(message);
    }
}
