package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import java.util.stream.DoubleStream;

/**
 * A basic decision tree classifier for use in machine learning coursework (6002B).
 */
public class CourseworkTree extends AbstractClassifier {

    /** Measure to use when selecting an attribute to split the data with. */
    private AttributeSplitMeasure attSplitMeasure = new IGAttributeSplitMeasure();

    /** Maximum depth for the tree. */
    private int maxDepth = Integer.MAX_VALUE;

    /** Minimum gain for a split to occur (reduces on depth)*/
    private double minGain = -1.0;

    /** Minimum Samples in order to produce a split */
    private int minSamples = -1;

    /** The root node of the tree. */
    private TreeNode root;

    @Override
    public String toString() {
        return "CourseworkTree {S:" + attSplitMeasure.getClass().getSimpleName() + ", Depth: " + maxDepth + ", Gain: " + minGain + ", Samples: " + minSamples + "}";
    }

    /**
     * Sets the attribute split measure for the classifier.
     *
     * @param attSplitMeasure the split measure
     */
    public void setAttSplitMeasure(AttributeSplitMeasure attSplitMeasure) {
        this.attSplitMeasure = attSplitMeasure;
    }

    /**
     * Sets the minimum gain for each node in the classifier.
     *
     * @param minGain the new minGain
     */
    public void setMinGain(double minGain) {
        this.minGain = minGain;
    }

    /**
     * Sets the minimum gain for each node in the classifier.
     *
     * @param minSamples the new minGain
     */
    public void setMinSamples(int minSamples) {
        this.minSamples = minSamples;
    }

    /**
     * Sets the max depth for the classifier.
     *
     * @param maxDepth the max depth
     */
    public void setMaxDepth(int maxDepth){
        this.maxDepth = maxDepth;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        //instances
        result.setMinimumNumberInstances(2);

        return result;
    }

    /**
     * Builds a decision tree classifier.
     *
     * @param data the training data
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes() - 1) {
            throw new Exception("Class attribute must be the last index.");
        }

        root = new TreeNode();
        root.buildTree(data, 0, minGain, minSamples);
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     */
    @Override
    public double classifyInstance(Instance instance) {
        double[] probs = distributionForInstance(instance);

        int maxClass = 0;
        for (int n = 1; n < probs.length; n++) {
            if (probs[n] > probs[maxClass]) {
                maxClass = n;
            }
        }

        return maxClass;
    }

    /**
     * Computes class distribution for instance using the decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     */
    @Override
    public double[] distributionForInstance(Instance instance) {
        return root.distributionForInstance(instance);
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        // SplitMeasure
        switch (Utils.getOption('S', options)) {
            case "chi-squared":
                setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
                break;
            case "gini":
                setAttSplitMeasure(new GiniAttributeSplitMeasure());
                break;
            case "info-gain":
                setAttSplitMeasure(new IGAttributeSplitMeasure());
                break;
            case "info-gain-ratio":
                IGAttributeSplitMeasure splitMeasure = new IGAttributeSplitMeasure();
                splitMeasure.useGain = true;
                setAttSplitMeasure(splitMeasure);
                break;
            default:
                //throw new Exception("Invalid option parameter -S " + Utils.getOption('S', options));
        };
        String dOptions = Utils.getOption('L', options);
        if (dOptions.length() > 0) {
            maxDepth = Integer.parseInt(dOptions);
        }
        String gOptions = Utils.getOption('G', options);
        if (gOptions.length() > 0) {
            minGain = Double.parseDouble(gOptions);
        }
        String mOptions = Utils.getOption('M', options);
        if (mOptions.length() > 0) {
            minSamples = Integer.parseInt(mOptions);
        }


    }

    /**
     * Class representing a single node in the tree.
     */
    private class TreeNode {

        /** Attribute used for splitting, if null the node is a leaf. */
        Attribute bestSplit = null;

        /** Best gain from the splitting measure if the node is not a leaf. */
        double bestGain = 0;

        /** Depth of the node in the tree. */
        int depth;

        /** The node's children if it is not a leaf. */
        TreeNode[] children;

        /** The class distribution if the node is a leaf. */
        double[] leafDistribution;

        /** The best numericalSplit for the dataset*/
        double bestNumericalSplit = -1.0;

        /** Minimum gain to produce a split */
        double minGain = -1;

        /**
         * Recursive function for building the tree.
         * Builds a single tree node, finding the best attribute to split on using a splitting measure.
         * Splits the best attribute into multiple child tree node's if they can be made, else creates a leaf node.
         *
         * @param data Instances to build the tree node with
         * @param depth the depth of the node in the tree
         * @param minimumGain minimum gain needed to produce a split
         * @param minimumSamples minimum instances needed to produce a split
         */
        void buildTree(Instances data, int depth, double minimumGain, int minimumSamples) throws Exception {
            this.depth = depth;
            this.minGain = minimumGain;

            // Loop through each attribute, finding the best one.
            for (int i = 0; i < data.numAttributes() - 1; i++) {

                if (minimumSamples != -1 && data.numInstances() < minimumSamples && bestSplit == null) { break; }
                //System.out.println("attribute " + (i+1) + "/" + data.numAttributes());
                double gain = attSplitMeasure.computeAttributeQuality(data, data.attribute(i));

                if (gain > bestGain && gain > minGain) {
                    bestSplit = data.attribute(i);
                    bestGain = gain;
                }
            }

            // If we found an attribute to split on, create child nodes.
            if (bestSplit != null) {
                Instances[] split = attSplitMeasure.splitData(data, bestSplit);
                children = new TreeNode[split.length];

                if (bestSplit.isNumeric()) { bestNumericalSplit = attSplitMeasure.getNumericSplitCriteria(data, bestSplit); };

                // Create a child for each value in the selected attribute, and determine whether it is a leaf or not.
                for (int i = 0; i < children.length; i++){
                    children[i] = new TreeNode();

                    boolean leaf = split[i].numDistinctValues(data.classIndex()) == 1 || depth + 1 == maxDepth;

                    if (split[i].isEmpty()) {
                        children[i].buildLeaf(data, depth + 1);
                    } else if (leaf) {
                        children[i].buildLeaf(split[i], depth + 1);
                    } else {
                        children[i].buildTree(split[i], depth + 1, minGain, minimumSamples);
                    }
                }
                // Else turn this node into a leaf node.
            } else {
                leafDistribution = classDistribution(data);
            }
        }

        /**
         * Builds a leaf node for the tree, setting the depth and recording the class distribution of the remaining
         * instances.
         *
         * @param data remaining Instances to build the leafs class distribution
         * @param depth the depth of the node in the tree
         */
        void buildLeaf(Instances data, int depth) {
            this.depth = depth;
            leafDistribution = classDistribution(data);
        }

        /**
         * Recursive function traversing node's of the tree until a leaf is found. Returns the leafs class distribution.
         *
         * @return the class distribution of the first leaf node
         */
        double[] distributionForInstance(Instance inst) {
            // If the node is a leaf return the distribution, else select the next node based on the best attributes
            // value.
            if (bestSplit == null) {
                return leafDistribution;
            } else {
                //System.out.println(this.getClass().getSimpleName() + "@depth = " + depth + " children = " +Arrays.toString(children));
                if (bestSplit.isNumeric()) { return children[inst.value(bestSplit) < bestNumericalSplit ? 0 : 1].distributionForInstance(inst); }
                //System.out.println(inst.numAttributes());
                return children[(int) inst.value(bestSplit)].distributionForInstance(inst);
            }
        }


        /**
         * Returns the normalised version of the input array with values summing to 1.
         *
         * @return the class distribution as an array
         */
        double[] classDistribution(Instances data) {
            double[] distribution = new double[data.numClasses()];
            for (Instance inst : data) {
                distribution[(int) inst.classValue()]++;
            }

            double sum = 0;
            for (double d : distribution){
                sum += d;
            }

            if (sum != 0){
                for (int i = 0; i < distribution.length; i++) {
                    distribution[i] = distribution[i] / sum;
                }
            }

            return distribution;
        }

        /**
         * Summarises the tree node into a String.
         *
         * @return the summarised node as a String
         */
        @Override
        public String toString() {
            String str;
            if (bestSplit == null){
                str = "Leaf," + Arrays.toString(leafDistribution) + "," + depth;
            } else {
                str = bestSplit.name() + "," + bestGain + "," + depth;
            }
            return str;
        }
    }

    /**
     * Main method.
     *
     * @param args the options for the classifier main
     */
    public static void main(String[] args) throws Exception {
        // Extra datasets: https://storm.cis.fordham.edu/~gweiss/data-mining/datasets.html
        String[] test_files = new String[]{
                "./src/main/java/ml_6002b_coursework/test_data/optdigits.arff",
                "./src/main/java/ml_6002b_coursework/test_data/Chinatown.arff"//,
        };

        Random rand = new Random();

        for (String file : test_files) {

            // Load data from file
            FileReader reader = new FileReader(file);
            Instances data = new Instances(reader);
            data.setClassIndex(data.numAttributes()-1);

            // Load Empty Test and Train set
            Instances[] train_test = new Instances[]{ new Instances(data, data.numInstances()), new Instances(data, data.numInstances())};

            // Randomly Separate Instances
            rand.setSeed(100243142);
            for (int i = 0; i<data.numInstances(); i++) {
                train_test[rand.nextInt(2)].add(data.instance(i));
            }

            // Initialise Information Gain Tree + Build Classifier from training data
            CourseworkTree IGTree = new CourseworkTree();
            IGTree.setOptions(new String[]{"-S", "info-gain"});
            IGTree.buildClassifier(new Instances(train_test[0]));

            // Initialise Information Gain Ratio Tree + Build Classifier from training data
            CourseworkTree IGRTree = new CourseworkTree();
            IGRTree.setOptions(new String[]{"-S", "info-gain-ratio"});
            IGRTree.buildClassifier(new Instances(train_test[0]));

            // Initialise Information Gini Impurity Measure Tree + Build Classifier from training data
            CourseworkTree GiniTree = new CourseworkTree();
            GiniTree.setOptions(new String[]{"-S", "gini"});
            GiniTree.buildClassifier(new Instances(train_test[0]));

            // Initialise Information Chi-Squared Tree + Build Classifier from training data
            CourseworkTree ChiTree = new CourseworkTree();
            ChiTree.setOptions(new String[]{"-S", "chi-squared"});
            ChiTree.buildClassifier(new Instances(train_test[0]));

            // Actual values of test set for accuracy calculation
            double[] actualValues = new double[train_test[1].numInstances()];
            for (int i = 0; i < actualValues.length; i++) {
                actualValues[i] = train_test[1].instance(i).classValue();
            }

            // 2D Result array [num_trees][test data count]
            double[][] results = new double[4][train_test[1].numInstances()];

            // 2D One hot encoding array for accuracy calculation
            double[][] one_hot = new double[4][train_test[1].numInstances()];

            // Bulk Classification from all 4 Tree split measures
            for (int i = 0; i < train_test[1].numInstances(); i++) {
                results[0][i] = IGTree.classifyInstance(train_test[1].instance(i));
                results[1][i] = IGRTree.classifyInstance(train_test[1].instance(i));
                results[2][i] = GiniTree.classifyInstance(train_test[1].instance(i));
                results[3][i] = ChiTree.classifyInstance(train_test[1].instance(i));
            }

            // Generate One Hot matrix from results
            for (int i = 0; i < results.length; i++) {
                for (int j = 0; j<results[i].length;j++) {
                    one_hot[i][j] = Double.compare(results[i][j], actualValues[j]) == 0 ? 1 : 0;
                }
            }

            //System.out.println(Arrays.toString(results[0]));
            //System.out.println(Arrays.toString(results[1]));
            //System.out.println(Arrays.toString(results[2]));
            //System.out.println(Arrays.toString(results[3]));
            //System.out.println(Arrays.toString(actualValues));

            System.out.println("DT using measure 'Information Gain' on '" + data.relationName() + "' problem has test accuracy = " + DoubleStream.of(one_hot[0]).sum() / actualValues.length * 100 + "%");
            System.out.println("DT using measure 'Information Gain Ratio' on '" + data.relationName() + "' problem has test accuracy = " + DoubleStream.of(one_hot[1]).sum() / actualValues.length * 100 + "%");
            System.out.println("DT using measure 'Gini Impurity Measure' on '" + data.relationName() + "' problem has test accuracy = " + DoubleStream.of(one_hot[2]).sum() / actualValues.length * 100 + "%");
            System.out.println("DT using measure 'Chi-Squared Statistic' on '" + data.relationName() + "' problem has test accuracy = " + DoubleStream.of(one_hot[3]).sum() / actualValues.length * 100 + "%");
            System.out.println();
        }
    }
}