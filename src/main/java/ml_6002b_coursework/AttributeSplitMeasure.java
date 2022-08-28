package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.util.Arrays;
import java.util.HashMap;

/**
 * Interface for alternative attribute split measures for Part 2.2 of the coursework
 */
public abstract class AttributeSplitMeasure {

    public abstract double computeAttributeQuality(Instances data, Attribute att) throws Exception;

    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
    public Instances[] splitData(Instances data, Attribute att) throws Exception {

        if (att.isNumeric()) {
            return this.splitDataOnNumeric(data, att);
        }


        Instances[] splitData = new Instances[att.numValues()];
        for (int i = 0; i < att.numValues(); i++) {
            splitData[i] = new Instances(data, data.numInstances());
        }

        for (Instance inst: data) {
            splitData[(int) inst.value(att)].add(inst);
        }

        for (Instances split : splitData) {
            split.compactify();
        }

        return splitData;
    }

    public double getNumericSplitCriteria(Instances data, Attribute att) throws Exception {

        //Prepare filter for numeric to nominal
        NumericToNominal filter = new NumericToNominal();
        filter.setInputFormat(data);
        filter.setAttributeIndicesArray(new int[]{att.index()});

        double bestGain = 0.0;
        double bestSplitCriteria = 0.0;
        for (Instance inst : data) {
            // For each instance use the instances value as the split criteria
            double newSplitCriteria = inst.value(att);
            Instances tempData = new Instances(data);
            // Discretize the data based on new split criteria
            for (Instance tempInst : tempData) {
                tempInst.setValue(att, (tempInst.value(att) <= newSplitCriteria ? 0 : 1));
            }
            //Filter the data into nominal
            filter.setInputFormat(tempData);
            tempData = Filter.useFilter(tempData, filter);
            // Calculate the gain
            double newGain = computeAttributeQuality(tempData, tempData.attribute(att.index()));
            // Compare to find if the new value is greater than or equal to our current best
            if (Double.compare(newGain, bestGain) >= 0) {
                bestGain = newGain;
                bestSplitCriteria = newSplitCriteria;
            } else { // VALUE LOWER == we have hit peak of curve
                break;
            }
        }

        return bestSplitCriteria;
    }

    public Instances[] splitDataOnNumeric(Instances data, Attribute att) throws Exception {
        Instances[] splitData = new Instances[]{new Instances(data, data.numInstances()), new Instances(data, data.numInstances())};

        // Sort instances bt attribute value
        data.sort(att);

        //Get the best split numeric criteria
        double bestSplitCriteria = getNumericSplitCriteria(data, att);

        for (Instance inst : data) {
            splitData[(int)inst.value(att) <= bestSplitCriteria ? 0 : 1].add(inst);
        }

        splitData[0].compactify();
        splitData[1].compactify();

        return splitData;
    }

}
