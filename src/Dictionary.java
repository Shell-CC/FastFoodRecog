import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.highgui.Highgui;

import java.io.*;
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

    private String name;
    private int size;
    private Mat centers;

    public List<Imagefeat> imagefeatList;

    public Mat getCenters() {
        return centers;
    }

    private static final TermCriteria termCriteria =
            new TermCriteria(TermCriteria.COUNT, 1000, 0.1);

    protected Dictionary(int size, final File folder) {
        this.size = size;
        this.imagefeatList = new ArrayList<Imagefeat>();
        // get codewords of all images
        getCodewordsOneRest(folder);
        // cluster into k and calculate bag of
        clusterCodewords(size);
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
    protected void getCodewordsOneInst(final File instFolder, int foodId) throws IOException{

        // Get background image first
        Mat background = new Mat();
        for (final File file : instFolder.listFiles()) {
            if (file.getName().equals("back.jpg")) {
                background = Highgui.imread(file.getPath());
            }
        }
        if (background.total() == 0) {
            throw new FileNotFoundException("Cannot find background image");
        }

        // extract surfs for each image and add to the list
        for (final File file : instFolder.listFiles()) {
            String filename = file.getPath();
            if (filename.endsWith(".jpg") && !filename.endsWith("back.jpg")) {
                FoodImage image = new FoodImage();
                image.read(filename);
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
        this.name = restFolder.getName();
        List<Mat> codewordsList = new ArrayList<Mat>();
        for (final File foodFolder : restFolder.listFiles()) {
            if (foodFolder.isDirectory()) {
                int foodId = Integer.parseInt(foodFolder.getName());
                for (final File instFolder : foodFolder.listFiles()) {
                    if (instFolder.isDirectory()) {
                        try {
                            getCodewordsOneInst(instFolder, foodId);
                        } catch (IOException e) {
                            System.out.println("Bad instance: " + instFolder.getPath());
                        }
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
        System.out.println("Clustering....");
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
        // seperate labels for each image and calculate bag of surfs
        System.out.println("Calculating bag of surfs");
        int start;
        int end = 0;
        for (Imagefeat imagefeat : imagefeatList) {
            start = end;
            end = end + imagefeat.getSurfs().rows();
            Mat labels = labelsAll.rowRange(start, end);
            BagOfSurf bagOfSurf = new BagOfSurf(size, labels);

            // save bag of surfs and add it to list
            imagefeat.setBagOfSurf(bagOfSurf);
        }
    }

    public void saveCenters(String path) {
        String filename = path + name + size + "Centers.txt";
        try {
            PrintWriter writer = new PrintWriter(filename);
            writer.print(centers.dump());
            writer.flush();
            writer.close();
        } catch (FileNotFoundException e) {
            throw new FileSystemNotFoundException("fail to write to " + filename);
        }
    }

    public void saveToDatabase(String path) {
        String filename = path + name + size + "Datas.csv";
        try {
            PrintWriter writer = new PrintWriter(filename);
            writer.println("FoodImage,NumOffeats,BagOfSurf,FoodId");
            for (Imagefeat imagefeat : imagefeatList) {
                writer.println(imagefeat.toString());
                writer.flush();
            }
            writer.close();
        } catch (FileNotFoundException e) {
            throw new FileSystemNotFoundException("fail to write to " + filename);
        }
    }

    public static Dictionary load(String filename) throws IOException{
        BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));
        String line = "";
        while ((line = bufferedReader.readLine()) != null) {
            String[] row;
            if (line.startsWith("[")) {
                row = line.substring(1, line.length()-1).split(",");
            } else {
                row = line.substring(0, line.length()-1).split(",");
            }
            for (String d : row) {
                System.out.println(Double.parseDouble(d));
            }
        }
        return instance;
    }
}

class Imagefeat {
    private String imgName;
    private Mat surfs;
    private BagOfSurf bagOfSurf;
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

    public int getFoodId() {
        return foodId;
    }

    public Imagefeat setFoodId(int foodId) {
        this.foodId = foodId;
        return this;
    }

    public BagOfSurf getBagOfSurf() {
        return bagOfSurf;
    }

    public Imagefeat setBagOfSurf(BagOfSurf bagOfSurf) {
        this.bagOfSurf = bagOfSurf;
        return this;
    }

    @Override
    public String toString() {
        return imgName + "," + surfs.rows() + ","
                + bagOfSurf.toString() + "," + foodId;
    }
}