import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.highgui.Highgui;

import java.io.File;
import java.nio.file.FileSystemNotFoundException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Shawn on 11/29/15.
 * Use classic singleton for the dictionary.
 *
 */
public class Dictionary {

    private static Dictionary instance = null;

    private int size;
    private Mat centers;
    private List<Mat> labelsList;

    private List<Imagefeat> imagefeatList;

    public List<Mat> getLabelsList() {
        return labelsList;
    }

    public Mat getCenters() {
        System.out.println(centers.size());
        return centers;
    }

    private static final TermCriteria termCriteria =
            new TermCriteria(TermCriteria.COUNT, 1000, 0.1);

    protected Dictionary(int size, final File folder) {
        this.size = size;
        this.imagefeatList = new ArrayList<Imagefeat>();
        getCodewordsOneRest(folder);
        clusterCodewords(size);
//        BagOfSurf hist = new BagOfSurf(size, labels, 0, labels.rows()-1);
    }


    /**
     * Build a specific dictionary for a specific restuarant.
     * Only one instance will generate for each restuarant
     * @param size Size of the dictionary.
     * @return The dictionary built.
     */
    public static Dictionary build(int size, final File folder) {
        if (instance == null) {
            instance = new Dictionary(size, folder);
        }
        return instance;
    }

    /**
     * Get all surf descriptors of all images in one instance.
     * @param instFolder The instance folder.
     */
    protected void getCodewordsOneInst(final File instFolder, int foodId) {

        // Get background image first
        Mat background = new Mat();
        for (final File file : instFolder.listFiles()) {
            if (file.getName().equals("back.jpg")) {
                background = Highgui.imread(file.getPath());
            }
        }
        if (background.total() == 0) {
            throw new FileSystemNotFoundException("Cannot find background image");
        }

        // extract surfs for each image and add to the list
        for (final File file : instFolder.listFiles()) {
            String filename = file.getPath();
            if (filename.endsWith(".jpg") && !filename.endsWith("back.jpg")) {
                FoodImage image = new FoodImage(filename);
                Mat mask = image.extractBackgroundMask(background);
                Mat surf = image.extractSurf(mask);
                System.out.println("Extracted " + surf.rows() + " SURFs for " + filename);

                // Save filename, surfs, foodId and add it to list
                Imagefeat imagefeat = new Imagefeat();
                imagefeat.setImgName(filename).setSurfs(surf).setFoodId(foodId);
                imagefeatList.add(imagefeat);
            }
        }
    }


    protected List<Mat> getCodewordsOneRest(final File restFolder) {
        List<Mat> codewordsList = new ArrayList<Mat>();
        for (final File foodFolder : restFolder.listFiles()) {
            if (foodFolder.isDirectory()) {
                int foodId = Integer.parseInt(foodFolder.getName());
                for (final File instFolder : foodFolder.listFiles()) {
                    if (instFolder.isDirectory()) {
                        getCodewordsOneInst(instFolder, foodId);
                    }
                }
            }
        }
        return codewordsList;
    }


    /**
     * Cluster all codewords into k clusters.
     * @param k Number of clusters
     * @return Labels(cluster number) of the input list of images.
     */
    protected void clusterCodewords(int k) {
        // merge all codewords into one mat
        Mat codewordsAll = new Mat();
        List<Mat> codewordsList = new ArrayList<Mat>(imagefeatList.size());
        for (Imagefeat imagefeat : imagefeatList) {
            codewordsList.add(imagefeat.getSurfs());
        }
        Core.vconcat(codewordsList, codewordsAll);
        Mat labelsAll = new Mat();
        Mat kmeansCenter = new Mat();
        Core.kmeans(codewordsAll, k, labelsAll, termCriteria, 3, Core.KMEANS_RANDOM_CENTERS, kmeansCenter);
        // save centers to dictionary
        this.centers = kmeansCenter;
        // seperate labels for each image and save to dictionary
        labelsList = new ArrayList<Mat>(codewordsList.size());
        int start;
        int end = 0;
        for (Mat m : codewordsList) {
            start = end;
            end = end + m.rows();
            labelsList.add(labelsAll.rowRange(start, end));
            assert m.size() == labelsList.get(labelsList.size()-1).size();
        }
        assert labelsList.size() == codewordsList.size();
    }
}

class Imagefeat {
    private String imgName;
    private Mat surfs;
    private Mat surfLabels;
    private int foodId;

    public String getImgName() {
        return imgName;
    }

    public Imagefeat setImgName(String imgName) {
        this.imgName = imgName;
        return this;
    }

    public Mat getSurfs() {
        return surfs;
    }

    public Imagefeat setSurfs(Mat surfs) {
        this.surfs = surfs;
        return this;
    }

    public Mat getSurfLabels() {
        return surfLabels;
    }

    public Imagefeat setSurfLabels(Mat surfLabels) {
        this.surfLabels = surfLabels;
        return this;
    }

    public int getFoodId() {
        return foodId;
    }

    public Imagefeat setFoodId(int foodId) {
        this.foodId = foodId;
        return this;
    }
}