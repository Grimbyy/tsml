package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RandomSubset;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.DoubleStream;

public class TreeEnsamble extends AbstractClassifier {

    /** Amount of tree's to store */
    int numTrees = 50;

    /** Proportion of attributes to select randomly */
    double attributeSelProportion = 0.5;

    /** Array of CourseworkTree */
    CourseworkTree[] ensemble;

    /** Array of Seeds used to select Subsets */
    int[] resampleSeeds;

    /** Seed for generation of ensemble */
    long ensambleSeed = -1;

    /** Filter used to Resample Data */
    //RandomSubsetFilter attFilter = new RandomSubsetFilter();
    RandomSubset attFilter = new RandomSubset();

    /** Use Majority voting or Average distributions */
    boolean averageDistributions = false;

    /***
     * Build each classifier stored within our ensamble
     * @param data set of instances serving as training data
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {

        // Initialise arrays
        ensemble = new CourseworkTree[numTrees];
        resampleSeeds = new int[numTrees];

        // Initialise Java Random
        Random rand = new Random();
        if (ensambleSeed != -1) { rand.setSeed(ensambleSeed); }

        attFilter.setNumAttributes(attributeSelProportion*data.numAttributes());

        for (int i = 0; i < ensemble.length; i++) {

            /*
             * Generate a seed used to select;
             *  1. the type of attribute split measure
             *  2. Maximum depth of tree
             *  3. Minimum gain criteria to perform a split
             *  4. Minimum samples needed @node to perform a split
             * */
            int randomSeed = rand.nextInt(100243142);
            Random optionRandom = new Random(randomSeed);

            resampleSeeds[i] = randomSeed;

            ensemble[i] = new CourseworkTree();

            // Select attribute measuring method
            switch (randomSeed/((100243142)/4)) {
                case 0:
                    ensemble[i].setOptions(new String[]{"-S", "chi-squared"});
                    break;
                case 1:
                    ensemble[i].setOptions(new String[]{"-S", "gini"});
                    break;
                case 2:
                    ensemble[i].setOptions(new String[]{"-S", "info-gain"});
                    break;
                case 3:
                    ensemble[i].setOptions(new String[]{"-S", "info-gain-ratio"});
                    break;
                default:
                    throw new Exception("Unknown error occured selecting tree options");
            }

            // Range for potential gain
            double[] minGainRange = new double[10];
            for (double j = 0; j < minGainRange.length; j++) {
                minGainRange[(int) j] = (j)/10.0;
            }

            // Range for min samples
            int[] minSamplesRange = new int[10];
            for (int j = 0; j < minGainRange.length; j++) {
                minSamplesRange[j] = (j*5);
            }

            // Range for maximum depth
            int[] maxDepthRange = new int[10];
            for (int j = 0; j < maxDepthRange.length; j++) {
                maxDepthRange[j] = (int) Math.pow(2, j);
            }

            // Best case you split on each attribute 3x
            ensemble[i].setMaxDepth(maxDepthRange[optionRandom.nextInt(maxDepthRange.length)]);

            // Minimum gain at the root of the tree (reduced after)
            ensemble[i].setMinGain(minGainRange[optionRandom.nextInt(minGainRange.length)]);

            // Minimum number of samples to produce split criteria
            ensemble[i].setMinSamples(minSamplesRange[optionRandom.nextInt(minSamplesRange.length)]);


            // RandomSubset filter to transform test data into format seen in buildClassifier via a seed.
            attFilter.setNumAttributes(attributeSelProportion*data.numAttributes());
            attFilter.setSeed(resampleSeeds[i]);
            attFilter.setInputFormat(data);

            // Filter data via a seed using Weka's RandomSubset filter
            //System.out.println("Building classifier " + (i+1) + "/" + numTrees);
            ensemble[i].buildClassifier(Filter.useFilter(data, attFilter));
            //System.out.println("Training Progress: " + (double)(i+1)/ensemble.length*100 + "%");
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] distributions = distributionForInstance(instance);

        // Get class index with highest score
        double highestDistribution = -1.0;
        int highestDistributionIndex = -1;
        for (int i = 0; i < distributions.length; i++) {
            if (distributions[i] > highestDistribution) {
                highestDistributionIndex = i;
                highestDistribution = distributions[i];
            }
        }

        // Return highest score index
        return highestDistributionIndex;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        // Temporarily store instance in an Instances (Compatibility with (RandomSubset)filtering method)
        Instances temp = new Instances(instance.dataset(), 0);
        temp.add(instance);

        if (averageDistributions) {
            double[] distributions = new double[instance.numClasses()];

            for (int i = 0; i < ensemble.length; i++) {

                // RandomSubset filter to transform test data into format seen in buildClassifier via a seed.
                attFilter.setNumAttributes(attributeSelProportion*instance.numAttributes());
                attFilter.setSeed(resampleSeeds[i]);
                attFilter.setInputFormat(temp);

                // Get distributions for instance filtered the exact same way the training data was using a seed.
                double[] newDistributions = ensemble[i].distributionForInstance(Filter.useFilter(temp, attFilter).instance(0));
                for (int j = 0; j < distributions.length; j++) {
                    distributions[j] += newDistributions[j];
                }
            }

            //double sum = DoubleStream.of(distributions).sum();
            for (int i = 0; i < distributions.length; i++) {
                distributions[i] = distributions[i] / numTrees;
            }

            return distributions;
        } else {
            double[] votes = new double[instance.numClasses()];

            for (int i = 0; i < ensemble.length; i++) {

                // RandomSubset filter to transform test data into format seen in buildClassifier via a seed.
                attFilter.setNumAttributes(attributeSelProportion*instance.numAttributes());
                attFilter.setSeed(resampleSeeds[i]);
                attFilter.setInputFormat(temp);

                int currentClassification = (int) ensemble[i].classifyInstance(Filter.useFilter(temp, attFilter).instance(0));

                votes[currentClassification] += 1;
            }


            //double sum = DoubleStream.of(votes).sum();
            for (int i = 0; i < votes.length; i++) {
                votes[i] = votes[i] / numTrees;
            }

            return votes;
        }
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        super.setOptions(options);
        switch (Utils.getOption('A', options).toLowerCase()) {
            case "true":
                this.averageDistributions = true;
                break;
            case "false":
                this.averageDistributions = false;
                break;
        }
        String nOption = Utils.getOption('N', options);
        if (nOption.length() > 0) {
            if (Double.parseDouble(nOption) > 0) {
                numTrees = Integer.parseInt(nOption);
            } else {
                System.err.println("Number of trees should be greater than zero");
            }
        }
        String pOption = Utils.getOption('P', options);
        if (pOption.length() > 0) {
            if (Double.parseDouble(pOption) > 0) {
                attributeSelProportion = Double.parseDouble(pOption);
            } else {
                System.err.println("Attribute Selection Proportion cannot be less than or equal to zero (stayed the same)");
            }
        }
        String sOption = Utils.getOption('S', options);
        if (sOption.length() > 0) {
            ensambleSeed = Integer.parseInt(sOption);
        }
    }

    public void setUseAverageDistributions(boolean useProp) {
        averageDistributions = useProp;
    }

    public void setAttributeSelProportion(double proportion) {
        this.attributeSelProportion = proportion;
    }

    public void setEnsambleSeed(long seed) {
        this.ensambleSeed = seed;
    }

    public void setNumTrees(int trees) {
        this.numTrees = trees;
    }

    @Override
    public String[] getOptions() {
        return super.getOptions();
    }

    @Override
    public String toString() {
        return super.toString();
    }

    public static void main(String[] args) throws Exception {

        // Extra datasets: https://storm.cis.fordham.edu/~gweiss/data-mining/datasets.html
        String[] test_files = new String[]{
                "./src/main/java/ml_6002b_coursework/test_data/optdigits.arff",
                "./src/main/java/ml_6002b_coursework/test_data/Chinatown.arff"
                //"./src/main/java/ml_6002b_coursework/test_data/UCI Continuous/pendigits/pendigits.arff"
        };

        Random rand = new Random();

        for (String file : test_files) {

            // Load data from file
            FileReader reader = new FileReader(file);
            Instances data = new Instances(reader);
            data.setClassIndex(data.numAttributes()-1);

            System.out.println("Problem: " + data.relationName());

            // Load Empty Test and Train set
            Instances[] train_test = new Instances[]{ new Instances(data, 0), new Instances(data, 0)};

            // Randomly Separate Instances
            // Done poorly, later figured out setupandrunexperiments splits data for me
            //rand.setSeed(100243142);
            for (int i = 0; i<data.numInstances(); i++) {
                train_test[rand.nextInt(2)].add(data.instance(i));
            }

            TreeEnsamble ensamble = new TreeEnsamble();
            ensamble.setUseAverageDistributions(true);
            ensamble.buildClassifier(new Instances(train_test[0]));

            // Actual values of test set for accuracy calculation
            double[] actualValues = new double[train_test[1].numInstances()];
            for (int i = 0; i < actualValues.length; i++) {
                actualValues[i] = train_test[1].instance(i).classValue();
            }

            // Result array [num_trees][test data count]
            double[] results = new double[train_test[1].numInstances()];

            // One hot encoding array for accuracy calculation
            double[] one_hot = new double[train_test[1].numInstances()];

            // Bulk Classification from all 4 Tree split measures
            for (int i = 0; i < train_test[1].numInstances(); i++) {
                if (i<5) {
                    System.out.println(Arrays.toString(ensamble.distributionForInstance(train_test[1].instance(i))));
                }
                results[i] = ensamble.classifyInstance(train_test[1].instance(i));
            }

            // Generate One Hot matrix from results
            for (int i = 0; i < results.length; i++) {
                    one_hot[i] = Double.compare(results[i], actualValues[i]) == 0 ? 1 : 0;
            }

            //System.out.println(Arrays.toString(results));
            //System.out.println(Arrays.toString(actualValues));

            System.out.println("Ensemble DT on '" + data.relationName() + "' problem has test accuracy = " + DoubleStream.of(one_hot).sum() / actualValues.length * 100 + "%");
            System.out.println();
        }

    }
}
