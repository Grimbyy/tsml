package ml_6002b_coursework;

import javax.xml.bind.SchemaOutputResolver;
import java.util.Arrays;

/**
 * Class for Part 2.1 of the coursework.
 */
public class AttributeMeasures {

    /**
     * Simple Log2 function
     * @param X any double
     * @return Log2 of X
     */
    private static double log2(double X){
        return X == 0 ? 0 : Math.log(X) / Math.log(2);
    }

    /***
     * Get the total amount of rows in present in the data
     * (I know its wildly inefficient to split up contents of for loops, chose readability over performance)
     * @param data a Contingency table
     * @return total amount of rows
     */
    private static int getDataTotal(int[][] data) {
        int totalData = 0;
        for (int i = 0; i<data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                // Loop through, count the amount of rows (in real dataset)
                totalData += data[i][j];
            }
        }

        return totalData;
    }

    /***
     * Get total events in a row/attribute
     * (I know its wildly inefficient to split up contents of for loops, chose readability over performance)
     * @param data Contingency table
     * @return 1D array containing totals for each row
     */
    private static int[] getTotalRow(int[][] data) {
        int[] attrTotals = new int[data.length];
        for (int i = 0; i<data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                // Sum the total of row/attribute value
                attrTotals[i] += data[i][j];
            }
        }

        return attrTotals;
    }

    /***
     * Get the amount of times each class appears within the data
     * (I know its wildly inefficient to split up contents of for loops, chose readability over performance)
     * @param data Contingency table
     * @param numClasses total number of classes
     * @return 1D array, one number for each class pertaining to their frequency
     */
    private static int[] getTotalClasses(int[][] data, int numClasses) {
        int[] classTotal = new int[numClasses];
        for (int i = 0; i<data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                // Sum the amount of times each class appears in data
                classTotal[j] += data[i][j];
            }
        }

        return classTotal;
    }

    /***
     * Get the probabilities of a Contingency table
     * (I know its wildly inefficient to split up contents of for loops, chose readability over performance)
     * @param data Contingency table
     * @return same structure as input Contingency table, replaced with % values
     */
    private static double[][] getProbabilities(int[][] data) {
        int[] rowTotals = getTotalRow(data);
        double[][] attrP = new double[data.length][data[0].length];
        for (int i = 0; i<data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                // Original value divided by the total amount in that row/attribute value
                attrP[i][j] = data[i][j] == 0 ? 0 : (double) data[i][j] / rowTotals[i];
            }
        }

        return attrP;
    }

    /***
     * Calculate Entropy for each row/attribute in a Contingency table
     * (I know its wildly inefficient to split up contents of for loops, chose readability over performance)
     * @param data Contingency table
     * @return 1D array containing the entropy for each row/attribute
     */
    private static double[] getEntropy(int[][] data) {
        double[] attrH = new double[data.length];
        double[][] probabilities = getProbabilities(data);

        for (int i = 0; i<data.length; i++) {
            for (int j = 0; j<probabilities[i].length; j++) {
                // Log 0 replaced with 0                                 Log 2 in java
                attrH[i] += probabilities[i][j] == 0 ? 0 : (probabilities[i][j]*log2(probabilities[i][j]));
            }
            attrH[i] = -attrH[i];
        }

        return attrH;
    }

    /***
     * Get the root entropy for a given Contingency table
     * (I know its wildly inefficient to split up contents of for loops, chose readability over performance)
     * @param data Contingency table
     * @param numClasses number of classes within the Contingency table
     * @return The entropy of the root
     */
    private static double getRootEntropy(int[][] data, int numClasses) {
        int[] classTotal = getTotalClasses(data, numClasses);
        int totalData = getDataTotal(data);
        double rootEntropy = 0.0;
        for (int i = 0; i<numClasses;i++){
            //System.out.println(classTotal[i] + "/" + totalData + "* log(" + classTotal[i] + "/" + totalData + ")");
            rootEntropy += ((double)classTotal[i]/totalData)*log2((double)classTotal[i]/totalData);
        }
        return -rootEntropy;
    }

    /***
     * Calculate the SplitInformation of a given Contingency table
     * (I know its wildly inefficient to split up contents of for loops, chose readability over performance)
     * @param data Contingency table
     * @return SplitInformation as a double
     */
    private static double getSplitInformation(int[][] data) {
        int totalEvents = getDataTotal(data);

        int[] totalOccurances = getTotalRow(data);

        double splitInformation = 0.0;
        for (int i = 0; i<totalOccurances.length;i++) {
            splitInformation += (double)totalOccurances[i]/totalEvents * log2((double)totalOccurances[i]/totalEvents);
        }

        return -splitInformation;
    }

    /***
     * Get the information gain from a Contingency table
     * @param data Contingency table
     * @return Information Gain statistic as a double
     */
    public static double measureInformationGain(int[][] data) {
        int numClasses = data[0].length;
        int numAttributeValues = data.length;

        // Amount of rows in data
        int totalData = getDataTotal(data);

        // Total Attributes to each row
        int[] attributeTotals = getTotalRow(data);

        //Attribute Entropy (Non-Root)
        double[] attributeEntropy = getEntropy(data);

        //Root entropy
        double rootEntropy = getRootEntropy(data, numClasses);

        //Calculate Weight
        double infoGain = 0.0;
        infoGain += rootEntropy;
        for (int i = 0; i<attributeTotals.length; i++) {
            infoGain -= ((double) attributeTotals[i]/(double)totalData)*attributeEntropy[i];
        }

        return infoGain;
    }

    /***
     * Get the information gain ratio statistic
     * @param data Contingency table
     * @return Information Gain Ratio statistic as a double
     */
    public static double measureInformationGainRatio(int[][] data) {
        double gain = measureInformationGain(data);
        double splitInformation = getSplitInformation(data);

        return gain/splitInformation;
    }

    /***
     * Get impurity score for split described in data
     * (I know its wildly inefficient to split up contents of for loops, chose readability over performance)
     * @param data Contingency table
     * @return 1D array of impurities from each row/attribute
     */
    private static double[] getImpurities(int[][] data) {

        double[][] probabilities = getProbabilities(data);
        double[] sumOfSquaredProbabilities = new double[probabilities.length];

        for (int i = 0; i<probabilities.length; i++) {
            sumOfSquaredProbabilities[i] = 1.0;
            for (int j = 0; j<probabilities[i].length;j++) {
                sumOfSquaredProbabilities[i] -= Math.pow(probabilities[i][j], 2);
            }
        }

        return sumOfSquaredProbabilities;
    }

    /***
     * get the impurity for a given Contingency table's root
     * (I know its wildly inefficient to split up contents of for loops, chose readability over performance)
     * @param data Contingency table
     * @return purity score of the Contingency table's root
     */
    private static double getRootImpurity(int[][] data) {
        int[] totalClasses = getTotalClasses(data, data[0].length);
        int totalRows = getDataTotal(data);

        double rootImpurity = 1.0;
        for (int i = 0; i<totalClasses.length;i++) {
            rootImpurity -= Math.pow((double)totalClasses[i]/totalRows, 2);
        }

        return rootImpurity;
    }

    /***
     * Get the gini impurity measure of a given Contingency table
     * (I know its wildly inefficient to split up contents of for loops, chose readability over performance)
     * @param data Contingency table
     * @return purity of an attribute in the form of a double
     */
    public static double measureGini(int[][] data) {
        double[] splitImpurities = getImpurities(data);

        double rootImpurity = getRootImpurity(data);

        double giniMeasure = rootImpurity;
        // Total rows in original table
        int rowCount = getDataTotal(data);
        // Totals per Contingency table row
        int[] rowTotals = getTotalRow(data);
        for (int i = 0; i<splitImpurities.length;i++) {
            giniMeasure -= ((double)rowTotals[i]/rowCount)*splitImpurities[i];
        }

        return giniMeasure;
    }

    /***
     * Get the probabilities at the root of the contingency table
     * (I know its wildly inefficient to split up contents of for loops, chose readability over performance)
     * @param data contingency table
     * @return 2D array containing root probabilities
     */
    private static double[] getRootProbabilities(int[][] data) {
        int numClasses = data[0].length;
        int[] classTotal = getTotalClasses(data, numClasses);
        int totalData = getDataTotal(data);

        double[] rootProbabilities = new double[classTotal.length];
        for (int i = 0; i<classTotal.length;i++) {
            rootProbabilities[i] = (double)classTotal[i]/totalData;
        }

        return rootProbabilities;
    }

    /***
     * Get the expected probabilities for a contingency table (chi-squared)
     * (I know its wildly inefficient to split up contents of for loops, chose readability over performance)
     * @param data contingency table
     * @return 2D array like the contingency table but the values replaced with probabilities
     */
    private static double[][] getExpectedProbabilities(int[][] data) {
        double[] rootProbabilities = getRootProbabilities(data);
        int[] rowTotals = getTotalRow(data);
        double[][] expectedProbabilities = new double[data.length][data[0].length];
        for (int i = 0; i<data.length; i++) {
            //System.out.println(Arrays.toString(data[i]) + " - " + rowTotals[i] + " * " + rootProbabilities[i] + " = " + (double) rowTotals[i] * rootProbabilities[i]);
            for (int j = 0; j < data[i].length;j++) {
                expectedProbabilities[i][j] = ((double) rowTotals[i] * rootProbabilities[j]);
            }
        }

        return expectedProbabilities;
    }

    /***
     * Measure the chi-squared of a given contingency table
     * @param data contingency table
     * @return Chi-squared statistic
     */
    public static double measureChiSquared(int[][] data) {
        double[][] expectedProbabilities = getExpectedProbabilities(data);

        double X = 0.0;
        for (int i = 0; i<data.length; i++) {
            for (int j = 0; j<data[i].length; j++) {
                //System.out.println("(" + data[i][j] + " - " + expectedProbabilities[i][j] + ")^2 / " + expectedProbabilities[i][j]);
                X += Math.pow((data[i][j] - expectedProbabilities[i][j]), 2) / expectedProbabilities[i][j];
            }
        }

        return X;
    }

    /**
     * Main method.
     *
     * @param args the options for the attribute measure main
     */
    public static void main(String[] args) {

        // Note: Root entropy is always -1

        int[][] peatyContingencyTable = new int[][]
                {
                        {4, 0},
                        {1, 5}
                };

        System.out.println("measure 'Information Gain' for Peaty = " + measureInformationGain(peatyContingencyTable));
        System.out.println("measure 'Gain Ratio' for Peaty = " + measureInformationGainRatio(peatyContingencyTable));
        System.out.println("measure 'Gini' for Peaty = " + measureGini(peatyContingencyTable));
        System.out.println("measure 'chi-squared' for Peaty = " + measureChiSquared(peatyContingencyTable));
    }

}
