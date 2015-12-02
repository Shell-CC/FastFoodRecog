import org.opencv.core.*;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.nio.file.FileSystemAlreadyExistsException;
import java.nio.file.FileSystemNotFoundException;

/**
 * Created by Shawn on 11/25/15.
 */

public class FoodImage {

    private Mat image;

    public FoodImage() {
        image = new Mat();
    }

    public FoodImage(String filename) {
        image = Highgui.imread(filename);
        if (image.total() == 0) {
            throw new FileSystemNotFoundException("Cannot find " + filename);
        }
    }

    public void write(String filename) {
        if (!Highgui.imwrite(filename, image)) {
            throw new FileSystemAlreadyExistsException(filename);
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
        return diff;
    }


    public Mat extractSurf(Mat mask) {
        MatOfKeyPoint features = new MatOfKeyPoint();
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

    private void drawFeatures(MatOfKeyPoint features) {
        Mat imageWithFeatures = new Mat();
        Features2d.drawKeypoints(image, features, imageWithFeatures);
        Highgui.imwrite("./res/KeyPoint.png", imageWithFeatures);
    }

}
