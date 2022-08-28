package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;

/***
 * Part 2.2.2
 */
public class ChiSquaredAttributeSplitMeasure extends AttributeSplitMeasure {

    @Override
    public int hashCode() {
        return 3;
    }

    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        if (data.attribute(att.index()) != att) {
            throw new Exception("Attribute is not found in the dataset");
        }

        Instances[] splitData = splitData(data, att);

        int[][] contingencyTable = new int[splitData.length][data.numClasses()];
        for (int i = 0; i < splitData.length; i++) {
            for (Instance inst : splitData[i]) {
                contingencyTable[i][(int) inst.classValue()] += 1;
            }
        }

        return AttributeMeasures.measureChiSquared(contingencyTable);
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) throws Exception {
        ChiSquaredAttributeSplitMeasure chiSplitMeasure = new ChiSquaredAttributeSplitMeasure();
        FileReader reader = new FileReader("./src/main/java/ml_6002b_coursework/test_data/Whiskey_Region.arff");
        Instances data = new Instances(reader);
        data.setClassIndex(data.numAttributes()-1);

        for (int i = 0; i<data.numAttributes()-1;i++) {
            System.out.println("measure 'Chi-Squared' for attribute '" + data.attribute(i).name() + "' splitting diagnosis = " + chiSplitMeasure.computeAttributeQuality(data, data.attribute(i)));
        }
    }
}
