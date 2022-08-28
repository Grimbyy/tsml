package ml_6002b_coursework;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RandomSubset;

import java.io.FileReader;

/**
 * lists of datasets available on blackboard for part 3 of the coursework.
 *
 * These problems have no missing values.
 */
public class DatasetLists {

    public static String[] nominalAttributeProblems={
            "balance-scale",
            //"car-evaluation",
            "chess-krvk",
            "chess-krvkp",
            "connect-4",
            "contraceptive-method",
            "fertility",
            "habermans-survival",
            "hayes-roth",
            "led-display",
            "lymphography",
            "molecular-promoters",
            "molecular-splice",
            "monks-1",
            "monks-2",
            "monks-3",
            "nursery",
            "optdigits",
            "pendigits",
            "semeion",
            "spect-heart",
            "tic-tac-toe",
            "zoo",
    };

    public static String[] continuousAttributeProblems={
            "bank",
            "blood",
            "breast-cancer-wisc-diag",
            "breast-tissue",
            "cardiotocography-10clases",
            "ecoli",
            "glass",
            "hill-valley",
            "image-segmentation",
            "ionosphere",
            "iris",
            "libras",
            "musk-2",
            "oocytes_merluccius_nucleus_4d",
            "oocytes_trisopterus_states_5b",
            "optical",
            "ozone",
            "page-blocks",
            "parkinsons",
            "pendigits",
            "planning",
            "post-operative",
            "ringnorm",
            "seeds",
            "spambase",
            "statlog-image",
            "statlog-landsat",
            "statlog-shuttle",
            "steel-plates",
            "synthetic-control",
            "twonorm",
            "vertebral-column-3clases",
            "wall-following",
            "waveform-noise",
            "wine-quality-white",
            "yeast",
    };


        public static void main(String[] args) throws Exception {
            String[] dataFiles = {
                    "./src/main/java/ml_6002b_coursework/test_data/optdigits.arff",
                    "./src/main/java/ml_6002b_coursework/test_data/Chinatown.arff"
            };

            for(String dataFile : dataFiles) {
                // Load data
                FileReader reader = new FileReader(dataFile);
                Instances data = new Instances(reader);
                data.setClassIndex(data.numAttributes() - 1);

                RandomSubset filter = new RandomSubset();
                filter.setNumAttributes(0.1);
                filter.setInputFormat(data);
                filter.setSeed(0);
                Instances filteredInstances = Filter.useFilter(data, filter);
                System.out.println(filteredInstances.toSummaryString());

                filter.setInputFormat(data);
                filter.setSeed(1255);
                filteredInstances = Filter.useFilter(data, filter);
                System.out.println(filteredInstances.toSummaryString());

            }
        }

}
