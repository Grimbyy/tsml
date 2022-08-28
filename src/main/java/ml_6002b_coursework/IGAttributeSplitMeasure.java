package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;

/***
 * Part 2.2.1
 */
public class IGAttributeSplitMeasure extends AttributeSplitMeasure {

    /** Controls whether to use Gain or Gain Ratio false = information gain, true = information gain ratio */
    public boolean useGain = false;

    @Override
    public int hashCode() {
        return useGain ? 1 : 0;
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

        return this.useGain ? AttributeMeasures.measureInformationGainRatio(contingencyTable) : AttributeMeasures.measureInformationGain(contingencyTable);
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) throws Exception {
        IGAttributeSplitMeasure igSplitMeasure = new IGAttributeSplitMeasure();
        FileReader reader = new FileReader("./src/main/java/ml_6002b_coursework/test_data/Whiskey_Region.arff");
        Instances data = new Instances(reader);
        data.setClassIndex(data.numAttributes()-1);

        for (int i = 0; i<data.numAttributes()-1;i++) {
            System.out.println("measure " + (igSplitMeasure.useGain ? "'Information Gain Ratio'" : "'Information Gain'") + " for attribute '" + data.attribute(i).name() + "' splitting diagnosis = " + igSplitMeasure.computeAttributeQuality(data, data.attribute(i)));
        }
        igSplitMeasure.useGain = !igSplitMeasure.useGain;
        for (int i = 0; i<data.numAttributes()-1;i++) {
            System.out.println("measure " + (igSplitMeasure.useGain ? "'Information Gain Ratio'" : "'Information Gain'") + " for attribute '" + data.attribute(i).name() + "' splitting diagnosis = " + igSplitMeasure.computeAttributeQuality(data, data.attribute(i)));
        }

    }

}
