package ml_6002b_coursework;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Arrays;
import java.util.Random;

/***
 * Implemented due to error with weka.filters.unsupervised.attribute.RandomSubset
 * whereby no matter the split % input, it defaults to 0.5 so will return half the attributes
 *
 * edit: I now realise some parameters have to be set BEFORE others when using the RandomSubset filter...
 */
public class RandomSubsetFilter {

    Random rand = new Random();

    double numAttributes = 0.5;

    public void setNumAttributes(double numAttributes) {
        this.numAttributes = numAttributes;
    }

    public double getNumAttributes() {
        return numAttributes;
    }

    public int getNumAttributes(Instances data) throws Exception {
        if (numAttributes >= 1.0) {
            return 0;
        } else {
            return (int) ((numAttributes)*(data.numAttributes()-1));
        }
    }

    public Instance filterInstance(Instance data, long seed) throws Exception {
        Instances temp = new Instances(data.dataset());
        temp.add(data);

        return filter(temp, seed).instance(0);
    }

    public Instances filter(Instances data, long seed) throws Exception {
        rand.setSeed(seed);

        int attCount = getNumAttributes(data);
        if (attCount == 0) { return data; }

        int[] indecies = new int[data.numAttributes()];
        int[] selectedIndecies = new int[attCount];

        for (int i = 0; i < attCount; i++) { // for each attribute we want in data
            while (selectedIndecies[i] == 0) {
                int location = rand.nextInt(data.numAttributes()) + 1;
                if (indecies[location-1] == 0 && location != data.numAttributes()) {
                    indecies[location-1] = 1;
                    selectedIndecies[i] = location;
                }
            }
        }

        for (int i = 0; i < selectedIndecies.length; i++) {
            selectedIndecies[i] -= 1;
        }

        Arrays.sort(selectedIndecies);
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(selectedIndecies);
        remove.setInvertSelection(false);
        remove.setInputFormat(data);

        return Filter.useFilter(data, remove);
    }
}
