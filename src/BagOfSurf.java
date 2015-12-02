import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by Shawn on 11/30/15.
 */
public class BagOfSurf {

    private int numOfFeats;
    private List<Double> hist;


    /**
     * Construct an empty histogram of surf of the image
     * @param numOfFeats The number of surfs in the image
     * @param size The length of the histogram.
     */
    public BagOfSurf(int numOfFeats, int size) {
        this.numOfFeats = numOfFeats;
        hist = new ArrayList<Double>(Collections.nCopies(size, 0.0));
    }

    /**
     * Construct the histogram of surfs from some range of the labels of surf descriptors
     * @param size The size of the histogram.
     * @param labels The labels of surf descriptors
     * @param from The from index of the labels (inclusive).
     * @param to The end index of the labels (exclusive).
     */
    public BagOfSurf(int size, Mat labels, int from, int to) {
        if (to <= from) throw new IllegalArgumentException("End index smaller than start index");
        this.numOfFeats = to - from;
        hist = new ArrayList<Double>(Collections.nCopies(size, 0.0));
        for (int i = from; i < to; i++) {
            int label = (int) (labels.get(i, 0)[0]);
            hist.set(label, hist.get(label) + 1);
        }
        for (int i = 0; i < hist.size(); i++) {
             hist.set(i, hist.get(i) / numOfFeats);
        }
    }

    /**
     * Construct the histogram of surfs from the labels of surf descriptors
     * @param size The size of the histogram.
     * @param labels The labels of surf descriptors
     */
    public BagOfSurf(int size, Mat labels) {
        this(size, labels, 0, labels.rows());
    }

    public List<Double> getHist() {
        return hist;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder("BagOfSurf{");
        if (hist.size() > 0) {
            builder.append(hist.get(0));
            for (int i = 1, N = hist.size(); i < N; i++) {
                builder.append(';').append(hist.get(i));
            }
        }
        builder.append('}');
        return builder.toString();
    }
}
