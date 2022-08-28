package ml_6002b_coursework;

import ResultsProcessing.CreateCriticalDifference;
import ResultsProcessing.MatlabController;
import ResultsProcessing.ResultTable;
import evaluation.MultipleEstimatorEvaluation;
import evaluation.PerformanceMetric;
import evaluation.ROCDiagramMaker;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterResults;
import evaluation.tuning.ParameterSpace;
import evaluation.tuning.Tuner;
import experiments.ClassifierExperiments;
import experiments.ExperimentalArguments;
import experiments.data.DatasetLoading;
import org.apache.commons.math3.geometry.euclidean.threed.Rotation;
import org.checkerframework.checker.units.qual.A;
import tsml.classifiers.kernel_based.ROCKETClassifier;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.LWL;
import weka.classifiers.meta.*;
import weka.classifiers.rules.M5Rules;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.*;
import weka.classifiers.trees.j48.Distribution;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import javax.crypto.spec.PSource;
import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Supplier;
import java.util.stream.DoubleStream;


public class Evaluation {

    /**
     * From DatasetLists.java:
     * 1. Decision Trees -
     *      i. Test difference in average accuracy for attribute selection methods (Task1 Task2)
     *      ii. Compare against Weka; ID3 & J48 classifiers
     * 2. Tree Ensemble vs Tree Tuning
     *      i. Test whether tuning CourseworkTree, including choosing attribute selection criteria method is better than ensembling
     *          i.a. Perform this experiment with proportion of attributes @ 100% then again at 50%
     * 3. Compare Ensemble against a range of built in Weka Classifiers, including other ensembles
     *      i.  random forest and rotation forest 4 sure
     * 4. Perform a case study on your assigned dataset to propose which classifier from those you have used would be best for this particular problem
     *      i. Use result data from 1, 2, 3. Best performing thing on datasets that look like the case study one form hypothesis
     *      ii. Experiment the exact same way as in 3. on new case study data
     *
     *
     * Notes:
     *      Include a table of all parameter settings of all classifiers in the paper.
     *      You can use both the discrete data and continuous data, although it may be advisable to split the analysis into two parts if you use both
     * */



    public static void main(String[] args) throws Exception {
        //getAttributeSelectionMethodOutputs();
        //compareAgainstID3J48();
        //getStatst1("./results/Task1-results/", "Task1", true);
        //treeTuning();
        //TreeTuningVsEnsembling();
        //getStats2("./results/Task2-results/", "Task2");
        //task7();
        //task72point0();
        //getTuneForCaseDataset();
        //task8(false);
        task8(true);
        //GetStats("./results/Task3-results/", "Task3");
        //GetStatst8();
        //makeDiagrams("Task3");
        //makeCaseDiagrams();
        //makeSummaryDiagramsTest("Task3");
        //makeCriticalDifferenceDiagrams("Task2");
        //makeCriticalDifferenceDiagrams("Task3");

        //dataStats();
        //caseStudyStats();
    }

    public static void caseStudyStats() throws Exception {
        Instances TEST = DatasetLoading.loadDataNullable("./src/main/java/ml_6002b_coursework/Evaluation/FiftyWords_TEST.arff");
        Instances TRAIN = DatasetLoading.loadDataNullable("./src/main/java/ml_6002b_coursework/Evaluation/FiftyWords_TRAIN.arff");
        double TESTAttributes = TEST.numAttributes();
        double TESTInstances = TEST.numInstances();
        double TESTClasses = TEST.numClasses();

        double TRAINAttributes = TRAIN.numAttributes();
        double TRAINInstances = TRAIN.numInstances();
        double TRAINClasses = TRAIN.numClasses();

        System.out.println(TRAINAttributes + " & " + TRAINInstances + " & " + TRAINClasses);
        System.out.println(TESTAttributes + " & " + TESTInstances + " & " + TESTClasses);
        System.out.println((TRAINInstances+TESTInstances));
        System.out.println("TRAIN/TEST & " + (TRAINInstances / (TRAINInstances+TESTInstances))*100 + "/" + (TESTInstances / (TRAINInstances+TESTInstances))*100 );
    }

    public static void dataStats() throws Exception {
        String[][] datasets = new String[][]{DatasetLists.nominalAttributeProblems, DatasetLists.continuousAttributeProblems};
        String[] categories = new String[]{"UCI Discrete", "UCI Continuous"};

        for (int x = 0; x < datasets.length; x++) {
            String dataCategory = categories[x];
            System.out.println(dataCategory);
            for (int y = 0; y < datasets[x].length; y++) {
                String dataset = datasets[x][y];

                Instances data = DatasetLoading.loadDataNullable("./src/main/java/ml_6002b_coursework/test_data/" + dataCategory + "/" + dataset + "/" + dataset + ".arff");
                int numAttributes = data.numAttributes()-1;
                int numInstances = data.numInstances();
                int classCount = data.numClasses();
                Distribution classDist = new Distribution(new Instances(data));
                double[] dists = new double[data.numClasses()];
                for (int i = 0; i < dists.length; i++) {
                    dists[i] = classDist.perClass(i);
                }
                double sum = DoubleStream.of(dists).sum();
                for (int i = 0; i < dists.length; i++) {
                    dists[i] = (dists[i] / sum) * 100.0;
                }
                int[] distInts = new int[dists.length];
                for (int i = 0; i < dists.length; i++) {
                    distInts[i] = (int) dists[i];
                }
                //bank & 0 & 0 & 0 & 0 \\
                //System.out.println(dataset.replaceAll("_", "\\\\_") + " & " + numAttributes + " & " + numInstances + " & " + classCount + " \\\\");
                boolean isEqual = true;
                int starting_number = distInts[0];
                for (int i = 0; i < distInts.length; i++) {
                    if (distInts[i] != starting_number) {
                        isEqual = false;
                        break;
                    }
                }
                System.out.println(
                        dataset.replaceAll("_", "\\\\_") + " & " +
                                (!isEqual ? Arrays.toString(distInts).replace('[', ' ').replaceAll("]", "").replaceAll(", ", "/") : Integer.toString(distInts[0]))
                        +" \\\\");
            }
        }
    }

    /**  i. Test difference in accuracy for attribute selection methods */
    public static void getAttributeSelectionMethodOutputs() throws Exception {
        Supplier<Classifier> IGRTree = () -> {
            CourseworkTree tree = new CourseworkTree();
            IGAttributeSplitMeasure sm = new IGAttributeSplitMeasure();
            sm.useGain = true;
            tree.setAttSplitMeasure(sm);
            return tree;
        };
        Supplier<Classifier> GiniTree = () -> {
            CourseworkTree tree = new CourseworkTree();
            GiniAttributeSplitMeasure sm = new GiniAttributeSplitMeasure();
            tree.setAttSplitMeasure(sm);
            return tree;
        };
        Supplier<Classifier> ChiTree = () -> {
            CourseworkTree tree = new CourseworkTree();
            ChiSquaredAttributeSplitMeasure sm = new ChiSquaredAttributeSplitMeasure();
            tree.setAttSplitMeasure(sm);
            return tree;
        };

        String[] classifierNames = new String[] {
                "InformationGain",
                "InformationGainRatio",
                "GiniTree",
                "ChiTree"
        };
        List<Supplier<Classifier>> classifierGenerators = Arrays.asList(
                () -> {return new CourseworkTree();},
                IGRTree,
                GiniTree,
                ChiTree
        );

        //String[][] datasets = new String[][]{DatasetLists.nominalAttributeProblems, DatasetLists.continuousAttributeProblems};
        String[][] datasets = new String[][]{new String[]{}, new String[]{"spambase","statlog-shuttle", "musk-2", "optical"}};
        String[] categories = new String[]{"UCI Discrete", "UCI Continuous"};
        // Remember to separate analysis of continuous and nominal datasets
        long seed = 0;

        for (int x = 0; x < datasets.length; x++) {
            String data_type = categories[x];

            ExperimentalArguments expArgs = new ExperimentalArguments();
            expArgs.dataReadLocation = "src/main/java/ml_6002b_coursework/test_data/"+data_type+"/";
            expArgs.resultsWriteLocation = "./results/Task1-results/";
            expArgs.forceEvaluation = true;

            ClassifierExperiments.setupAndRunMultipleExperimentsThreaded(expArgs, classifierNames, classifierGenerators, datasets[x], 0, 1);
            System.out.println("DONE! THANK GOD!");
        }
    }

    public static void getStatst1(String resultsPath, String experimentName, boolean accuracyOnly) throws Exception {

        File resPathFile = new File(resultsPath);
        String[] classifiers = new String[resPathFile.listFiles().length];
        int i = 0;
        for (File folder : resPathFile.listFiles()) {
            classifiers[i] = folder.getName();
            i++;
        }
        ArrayList<PerformanceMetric> metrics;
        if (accuracyOnly) {
            metrics = new ArrayList<>();
            metrics.add(PerformanceMetric.acc);
        } else {
            metrics = (ArrayList<PerformanceMetric>) PerformanceMetric.getAllPredictionStatistics();
        }


        new MultipleEstimatorEvaluation("./analysis", experimentName + "-nominal", 1)
                .setTestResultsOnly(true)
                .setUseAccuracyOnly()
                .setDatasets(DatasetLists.nominalAttributeProblems)
                .readInEstimators(classifiers, classifiers, resultsPath)
                .runComparison();

        String[] classifiersbarid3 = new String[resPathFile.listFiles().length-1];
        i = 0;
        for (File folder : resPathFile.listFiles()) {
            if (!folder.getName().equals("Id3")) {
                classifiersbarid3[i] = folder.getName();
                i++;
            }
        }

        //Continuous Ignore id3
        new MultipleEstimatorEvaluation("./analysis", experimentName + "-continuous", 1)
                .setTestResultsOnly(true)
                .setUseAccuracyOnly()
                .setDatasets(DatasetLists.continuousAttributeProblems)
                .readInEstimators(classifiersbarid3, classifiersbarid3, resultsPath)
                .runComparison();

        /*ArrayList<double[][]> results = new ArrayList<>();
        ArrayList<double[][]> totals = new ArrayList<>();
        ArrayList<String[][]> names = new ArrayList<>();

        int treeCounter = 0;
        for (File tree : resultsRoot.listFiles()) {
            results.add(new double[tree.listFiles().length][2]);
            totals.add(new double[tree.listFiles().length][2]);
            names.add(new String[tree.listFiles().length][2]);
            int treeTypeCounter = 0;
            for (File treeType : tree.listFiles()) {
                int dataTypeCounter = 0;
                for (File dataType : treeType.listFiles()) {
                    File predictions = new File(dataType.getAbsolutePath() + "/Predictions");
                    for (File predictionFolder : predictions.listFiles()) {
                        for (File prediction : predictionFolder.listFiles()) {
                            results.get(treeCounter)[treeTypeCounter][dataTypeCounter] += new ClassifierResults(prediction.getAbsolutePath()).getAcc();
                            totals.get(treeCounter)[treeTypeCounter][dataTypeCounter] += 1;
                        }
                    }
                    names.get(treeCounter)[treeTypeCounter][dataTypeCounter] = tree.getName() +"["+ treeType.getName() + "]" + " for " + dataType.getName();
                    dataTypeCounter++;
                }
                treeTypeCounter++;
            }
            treeCounter++;
        }

        for (int x = 0; x < results.size(); x++) {
            for (int y = 0; y < results.get(x).length; y++) {
                for (int z = 0; z < results.get(x)[y].length; z++) {
                    results.get(x)[y][z] = (results.get(x)[y][z] / totals.get(x)[y][z]) * 100.0;
                }
            }
        };

        for (int x = 0; x < results.size(); x++) {
            for (int y = 0; y < results.get(x).length; y++) {
                for (int z = 0; z < results.get(x)[y].length; z++) {
                    System.out.println(names.get(x)[y][z] + " = " + results.get(x)[y][z]);
                }
            }
        };*/
    }

    /**  Compare against Weka; ID3 & J48 classifiers */
    public static void compareAgainstID3J48() throws Exception {

        String[] classifierNames = new String[] {
                "Id3",
                "J48"
        };
        List<Supplier<Classifier>> classifierGenerators = Arrays.asList(
                () -> {return new Id3();},
                () -> {return new J48();}
        );

        String[][] datasets = new String[][]{DatasetLists.nominalAttributeProblems, DatasetLists.continuousAttributeProblems};
        //String[][] datasets = new String[][]{new String[]{}, new String[]{"spambase","statlog-shuttle", "musk-2", "optical"}};
        String[] categories = new String[]{"UCI Discrete", "UCI Continuous"};
        // Remember to separate analysis of continuous and nominal datasets
        long seed = 0;

        for (int x = 0; x < datasets.length; x++) {
            String data_type = categories[x];

            ExperimentalArguments expArgs = new ExperimentalArguments();
            expArgs.dataReadLocation = "src/main/java/ml_6002b_coursework/test_data/"+data_type+"/";
            expArgs.resultsWriteLocation = "./results/Task1-results/";
            expArgs.forceEvaluation = true;

            ClassifierExperiments.setupAndRunMultipleExperimentsThreaded(expArgs, classifierNames, classifierGenerators, datasets[x], 0, 1);
            System.out.println("DONE! THANK GOD!");
        }
    }

    public static void treeTuning() throws Exception {

        // Tuner printing to console is annoying
        PrintStream oldStream = System.out;
        PrintStream dummyStream = new PrintStream(new OutputStream() {
            @Override
            public void write(int b) throws IOException {
                //nothing
            }
        });
        System.setOut(dummyStream);

        int seed = 0;

        CourseworkTree tuningTree = new CourseworkTree();
        Tuner tuner = new Tuner();
        //tuner.setPathToSaveParameters("./results/Tuning/CourseworkTree/");
        tuner.setSeed(seed);

        //int[][] splitmeasurevotes = new int[2][4];

        double[] minGainRange = new double[10];
        for (double i = 0; i < minGainRange.length; i++) {
            minGainRange[(int) i] = (i)/10.0;
        }

        oldStream.println(Arrays.toString(minGainRange));

        int[] minSamplesRange = new int[10];
        for (int i = 0; i < minGainRange.length; i++) {
            minSamplesRange[i] = (i*5);
        }

        oldStream.println(Arrays.toString(minSamplesRange));

        int[] maxDepthRange = new int[10];
        for (int i = 0; i < maxDepthRange.length; i++) {
            maxDepthRange[i] = (int) Math.pow(2, i);
        }

        oldStream.println(Arrays.toString(maxDepthRange));

        String[] splitMeasures = new String[]{"info-gain", "info-gain-ratio", "gini", "chi-squared"};
        ParameterSpace space = new ParameterSpace();
        space.addParameter("S", splitMeasures);
        space.addParameter("L", maxDepthRange);
        space.addParameter("G", minGainRange);
        space.addParameter("M", minSamplesRange);

        String[] nominalDataSets = new String[]{"hayes-roth", "fertility", "tic-tac-toe"};
        String[] continuiousDataSets = new String[]{"blood", "wine-quality-white", "seeds"};

        String[][] datasets = new String[][]{DatasetLists.nominalAttributeProblems, DatasetLists.continuousAttributeProblems};
        String[] datasetType = new String[]{"UCI Discrete", "UCI Continuous"};

        String[] resultsString = new String[datasets[0].length+datasets[1].length];

        ArrayList<ArrayList<String>> usedDatasets = new ArrayList<>();
        usedDatasets.add(new ArrayList<>());
        usedDatasets.add(new ArrayList<>());

        for (int x = 0; x < datasets.length-1; x++) {
            String dataCategory = datasetType[x];
            for (int y = 0; y < datasets[x].length; y++) {
                String dataset = datasets[x][y];

                Instances data = DatasetLoading.loadDataNullable("./src/main/java/ml_6002b_coursework/test_data/"+dataCategory+"/"+dataset+"/"+dataset+".arff");

                double perc = (y / datasets[0].length) * 100.0;

                if (data.numInstances() < 1000) {
                    oldStream.println("["+perc + "%] Tuning for " + dataset + "["+data.numAttributes()+","+data.numInstances()+"]...");
                    Instances[] splitData = InstanceTools.resampleInstances(data, seed, 0.5);

                    usedDatasets.get(x).add(dataset);

                    ParameterResults results = tuner.tune(tuningTree, splitData[0], space);

                    //oldStream.println(dataset + " done! results = " + results.paras.toString() + "");
                    resultsString[y] = dataset + "," + results.paras.getParameterValue("S") + "," + results.paras.getParameterValue("L") + "," + results.paras.getParameterValue("G") + "," + results.paras.getParameterValue("M");
                } else {
                    oldStream.println("["+perc + "%] Dataset " + dataset + "["+data.numAttributes()+","+data.numInstances()+ "] is too large! moving on");
                }
            }

        }

        for (int i = 0; i < resultsString.length; i++) {
            oldStream.println(resultsString[i]);
        }

        for (ArrayList<String> l : usedDatasets) {
            for (String d : l) {
                oldStream.print("\"" + d + "\",");
            }
            oldStream.println();
        }
    }

    public static void TreeTuningVsEnsembling() throws Exception {
        int seed = 0;

        //String[] nominal_datasets = new String[]{"balance-scale","fertility","habermans-survival","hayes-roth","led-display","lymphography","molecular-promoters","monks-1","monks-2","monks-3","spect-heart","tic-tac-toe","zoo"};
        File csv = new File("./results/Tuning/bestNominalTunes.csv");
        Scanner tunes = new Scanner(csv);

        String[] headers = tunes.nextLine().split(",");
        System.out.println(Arrays.toString(headers));

        ExperimentalArguments expArgsTuned = new ExperimentalArguments();
        expArgsTuned.estimatorName = "Task2-results/CourseworkTreeTuned";

        ExperimentalArguments expArgsEnsamble100 = new ExperimentalArguments();
        expArgsEnsamble100.estimatorName = "Task2-results/CourseworkTreeEnsemble100";

        ExperimentalArguments expArgsEnsamble50 = new ExperimentalArguments();
        expArgsEnsamble50.estimatorName = "Task2-results/CourseworkTreeEnsemble50";

        ExperimentalArguments expArgsEnsamble100WDist = new ExperimentalArguments();
        expArgsEnsamble100WDist.estimatorName = "Task2-results/CourseworkTreeEnsemble100WDist";

        ExperimentalArguments expArgsEnsamble50WDist = new ExperimentalArguments();
        expArgsEnsamble50WDist.estimatorName = "Task2-results/CourseworkTreeEnsemble50WDist";

        while (tunes.hasNext()) {
            String[] tuneArgs = tunes.nextLine().split(",");

            expArgsTuned.datasetName = tuneArgs[Arrays.asList(headers).indexOf("dataset")];
            expArgsEnsamble100.datasetName = tuneArgs[Arrays.asList(headers).indexOf("dataset")];
            expArgsEnsamble50.datasetName = tuneArgs[Arrays.asList(headers).indexOf("dataset")];
            expArgsEnsamble100WDist.datasetName = tuneArgs[Arrays.asList(headers).indexOf("dataset")];
            expArgsEnsamble50WDist.datasetName = tuneArgs[Arrays.asList(headers).indexOf("dataset")];


            CourseworkTree tree = new CourseworkTree();
            String[] options = new String[]{
                    "S", tuneArgs[Arrays.asList(headers).indexOf("s")],
                    "L", tuneArgs[Arrays.asList(headers).indexOf("l")],
                    "G", tuneArgs[Arrays.asList(headers).indexOf("g")],
                    "M", tuneArgs[Arrays.asList(headers).indexOf("m")]
            };
            tree.setOptions(options);

            TreeEnsamble treeEnsamble100 = new TreeEnsamble();
            treeEnsamble100.setEnsambleSeed(seed);
            treeEnsamble100.setAttributeSelProportion(1.0);
            TreeEnsamble treeEnsamble50 = new TreeEnsamble();
            treeEnsamble50.setEnsambleSeed(seed);

            TreeEnsamble treeEnsamble100WDist = new TreeEnsamble();
            treeEnsamble100WDist.setEnsambleSeed(seed);
            treeEnsamble100WDist.setAttributeSelProportion(1.0);
            treeEnsamble100WDist.setUseAverageDistributions(true);

            TreeEnsamble treeEnsamble50WDist = new TreeEnsamble();
            treeEnsamble50WDist.setEnsambleSeed(seed);
            treeEnsamble50WDist.setUseAverageDistributions(true);

            expArgsTuned.classifier = tree;
            expArgsEnsamble100.classifier = treeEnsamble100;
            expArgsEnsamble50.classifier = treeEnsamble50;
            expArgsEnsamble100WDist.classifier = treeEnsamble100WDist;
            expArgsEnsamble50WDist.classifier = treeEnsamble50WDist;

            ClassifierExperiments.buildExperimentDirectoriesAndFilenames(expArgsTuned, tree);
            ClassifierExperiments.buildExperimentDirectoriesAndFilenames(expArgsEnsamble100, treeEnsamble100);
            ClassifierExperiments.buildExperimentDirectoriesAndFilenames(expArgsEnsamble50, treeEnsamble50);
            ClassifierExperiments.buildExperimentDirectoriesAndFilenames(expArgsEnsamble100WDist, treeEnsamble100WDist);
            ClassifierExperiments.buildExperimentDirectoriesAndFilenames(expArgsEnsamble50WDist, treeEnsamble50WDist);

            Instances data = DatasetLoading.loadDataNullable("./src/main/java/ml_6002b_coursework/test_data/UCI Discrete/"+expArgsTuned.datasetName+"/"+expArgsTuned.datasetName+".arff");
            Instances[] splitData = InstanceTools.resampleInstances(data, seed, 0.5);

            ClassifierExperiments.runExperiment(expArgsTuned, new Instances(splitData[0]), new Instances(splitData[1]), tree);
            ClassifierExperiments.runExperiment(expArgsEnsamble100, new Instances(splitData[0]), new Instances(splitData[1]), treeEnsamble100);
            ClassifierExperiments.runExperiment(expArgsEnsamble50, new Instances(splitData[0]), new Instances(splitData[1]), treeEnsamble50);
            ClassifierExperiments.runExperiment(expArgsEnsamble100WDist, new Instances(splitData[0]), new Instances(splitData[1]), treeEnsamble100WDist);
            ClassifierExperiments.runExperiment(expArgsEnsamble50WDist, new Instances(splitData[0]), new Instances(splitData[1]), treeEnsamble50WDist);

        }
    }

    public static void getStats2(String resultsPath, String taskName) throws Exception {
        File resPathFile = new File(resultsPath);
        String[] classifiers = new String[resPathFile.listFiles().length];
        int i = 0;
        for (File folder : resPathFile.listFiles()) {
            classifiers[i] = folder.getName();
            i++;
        }
        
        String[] datasets = new String[]{"balance-scale","fertility","habermans-survival","hayes-roth","led-display","lymphography","molecular-promoters","monks-1","monks-2","monks-3","spect-heart","tic-tac-toe","zoo"};

        new MultipleEstimatorEvaluation("./analysis", taskName, 1)
                .setTestResultsOnly(true)
                .setUseAllStatistics()
                .setDatasets(datasets)
                .readInEstimators(classifiers, classifiers, resultsPath)
                .runComparison();
    }

    /** Compare your ensemble against a range of built in Weka classifiers, including other en-
     sembles, on the provided classification problems */
    public static void task7() throws Exception {
        //random forest and rotation forest 4 sure
        Supplier<Classifier> treeEnsembleSupplier = () -> {
            TreeEnsamble tree = new TreeEnsamble();
            tree.setUseAverageDistributions(true);
            tree.setAttributeSelProportion(1.0);
            return tree;
        };

        String[] classifierNames = new String[] {
              //"RotationForrest",
              //"RandomForrest",
              //"AdaBoostM1",
              //"Vote",
              //"Stacking",
              //"IBk",
              //"J48",
              //"PART",
              //"NaiveBayes",
              //"OneR",
              //"SMO",
              ////"Logistic",
              //"LogitBoost",
              "TreeEnsemble"
        };
        List<Supplier<Classifier>> classifierGenerators = Arrays.asList(
                //() -> {return new RotationForest();},
                //() -> {return new RandomForest()  ;},
                //() -> {return new AdaBoostM1()    ;},
                //() -> {return new Vote()          ;},
                //() -> {return new Stacking()      ;},
                //() -> {return new IBk()           ;},
                //() -> {return new J48()           ;},
                //() -> {return new PART()          ;},
                //() -> {return new NaiveBayes()    ;},
                //() -> {return new OneR()          ;},
                //() -> {return new SMO()           ;},
                //() -> {return new Logistic()      ;},
                //() -> {return new LogitBoost()    ;},
                treeEnsembleSupplier
        );

        String[][] datasets = new String[][]{DatasetLists.nominalAttributeProblems, DatasetLists.continuousAttributeProblems};
        //String[][] datasets = new String[][]{new String[]{"connect-4"}, new String[]{}};
        String[] categories = new String[]{"UCI Discrete", "UCI Continuous"};
        // Remember to separate analysis of continuous and nominal datasets
        long seed = 0;

        for (int x = 0; x < datasets.length; x++) {
            String data_type = categories[x];

            ExperimentalArguments expArgs = new ExperimentalArguments();
            expArgs.dataReadLocation = "src/main/java/ml_6002b_coursework/test_data/"+data_type+"/";
            expArgs.resultsWriteLocation = "./results/Task3-results/";
            expArgs.forceEvaluation = true;

            ClassifierExperiments.setupAndRunMultipleExperimentsThreaded(expArgs, classifierNames, classifierGenerators, datasets[x], 0, 1);
            System.out.println("DONE! THANK GOD!");
        }
    }

    //Force statlog shuttle to work
    public static void task72point0() {
        TreeEnsamble tree = new TreeEnsamble();
        tree.setUseAverageDistributions(true);
        tree.setAttributeSelProportion(1.0);

        Instances data = DatasetLoading.loadDataNullable("src/main/java/ml_6002b_coursework/test_data/UCI Continuous/statlog-shuttle/statlog-shuttle.arff");
        Instances[] split = InstanceTools.resampleInstances(data, 0, 0.5);

        System.out.println("Instances resampled");

        ExperimentalArguments expArgs = new ExperimentalArguments();
        expArgs.estimatorName = "Task3-results/TreeEnsemble";
        expArgs.datasetName = "statlog-shuttle";
        expArgs.classifier = tree;

        System.out.println("Experiment Started");
        ClassifierExperiments.runExperiment(expArgs, new Instances(split[0]), new Instances(split[1]), tree);
        System.out.println("Experiment Complete");
    }

    public static void getTuneForCaseDataset() throws Exception {
        int seed = 0;

        CourseworkTree tuningTree = new CourseworkTree();
        Tuner tuner = new Tuner();
        //tuner.setPathToSaveParameters("./results/Tuning/CourseworkTree/");
        tuner.setSeed(seed);

        double[] minGainRange = new double[10];
        for (double i = 0; i < minGainRange.length; i++) {
            minGainRange[(int) i] = (i)/10.0;
        }


        int[] minSamplesRange = new int[10];
        for (int i = 0; i < minGainRange.length; i++) {
            minSamplesRange[i] = (i*5);
        }


        int[] maxDepthRange = new int[10];
        for (int i = 0; i < maxDepthRange.length; i++) {
            maxDepthRange[i] = (int) Math.pow(2, i);
        }

        String[] splitMeasures = new String[]{"info-gain", "info-gain-ratio", "gini", "chi-squared"};
        ParameterSpace space = new ParameterSpace();
        space.addParameter("S", splitMeasures);
        space.addParameter("L", maxDepthRange);
        space.addParameter("G", minGainRange);
        space.addParameter("M", minSamplesRange);

        Instances TRAIN = DatasetLoading.loadDataNullable("./src/main/java/ml_6002b_coursework/Evaluation/FiftyWords_TRAIN.arff");
        Discretize toNominal = new Discretize();
        toNominal.setInputFormat(TRAIN);
        ParameterResults results = tuner.tune(tuningTree, Filter.useFilter(TRAIN, toNominal), space);
        //results = {S: info-gain-ratio, G: 0.0, L: 16, M: 0, }
        System.out.println("results = " + results.paras.toString() + "");
    }

    /** 4. Perform a case study on your assigned dataset to propose which classifier
     * from those you have used would be best for this particular problem */
    public static void task8(boolean makeNominal) throws Exception {
        String[] classifierNames;
        Classifier[] classifiers;

        classifierNames = new String[]{
                "CourseworkTreeEnsemble100",
                "CourseworkTreeEnsemble50",
                "CourseworkTreeEnsemble100Dist",
                "CourseworkTreeEnsemble50Dist",
                "CourseworkTreeInfoGain",
                "CourseworkTreeInfoGainRatio",
                "CourseworkTreeGini",
                "CourseworkTreeChiSquared",
                "CourseworkTreeTuned",
                //Other things (remove where needed)
                "RotationForrest",
                "RandomForrest",
                "AdaBoostM1",
                "Vote",
                "Stacking",
                "IBk",
                "J48",
                "PART",
                "NaiveBayes",
                "OneR",
                "SMO",
                "LogitBoost"
        };

        //"CourseworkTreeEnsemble100"
        TreeEnsamble CTEO = new TreeEnsamble();
        CTEO.setAttributeSelProportion(1.0);
        //"CourseworkTreeEnsemble50"
        TreeEnsamble CTEF = new TreeEnsamble();
        //"CourseworkTreeEnsemble100Dist"
        TreeEnsamble CTEOD = new TreeEnsamble();
        CTEOD.setAttributeSelProportion(1.0);
        CTEOD.setUseAverageDistributions(true);
        //"CourseworkTreeEnsemble50Dist"
        TreeEnsamble CTEFD = new TreeEnsamble();
        CTEFD.setAttributeSelProportion(1.0);
        CTEFD.setUseAverageDistributions(true);
        //"CourseworkTreeInfoGain"
        CourseworkTree CTIG = new CourseworkTree();
        //"CourseworkTreeInfoGainRatio"
        CourseworkTree CTIGR = new CourseworkTree();
        IGAttributeSplitMeasure useRatio = new IGAttributeSplitMeasure();
        useRatio.useGain = true;
        CTIGR.setAttSplitMeasure(useRatio);
        //"CourseworkTreeGini"
        CourseworkTree CTG = new CourseworkTree();
        CTG.setAttSplitMeasure(new GiniAttributeSplitMeasure());
        //"CourseworkTreeChiSquared"
        CourseworkTree CTCS = new CourseworkTree();
        CTCS.setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());
        //"CourseworkTreeTuned" {S: info-gain-ratio, G: 0.0, L: 16, M: 0, }
        CourseworkTree CTT = new CourseworkTree();
        String[] options = new String[]{
                "-S", "info-gain-ratio",
                "-G", "0.0",
                "-L", "16",
                "-M", "0"
        };
        CTT.setOptions(options);

        classifiers = new Classifier[] {
                CTEO,
                CTEF,
                CTEOD,
                CTEFD,
                CTIG,
                CTIGR,
                CTG,
                CTCS,
                CTT,
                new RotationForest(),
                new RandomForest(),
                new AdaBoostM1(),
                new Vote(),
                new Stacking(),
                new IBk(),
                new J48(),
                new PART(),
                new NaiveBayes(),
                new OneR(),
                new SMO(),
                new LogitBoost()
        };

        if (classifiers.length != classifierNames.length) { throw new Exception("Classifiers dont match up to namelist!"); }

        Instances TRAIN = DatasetLoading.loadDataNullable("./src/main/java/ml_6002b_coursework/Evaluation/FiftyWords_TRAIN.arff");
        Instances TEST = DatasetLoading.loadDataNullable("./src/main/java/ml_6002b_coursework/Evaluation/FiftyWords_TEST.arff");

        if (makeNominal) {
            Discretize filter = new Discretize();
            filter.setInputFormat(TRAIN);
            TRAIN = Filter.useFilter(TRAIN, filter);
            TEST = Filter.useFilter(TEST, filter);
        }

        for (int c = 0; c < classifiers.length; c++) {
            ExperimentalArguments expArgs = new ExperimentalArguments();
            expArgs.estimatorName = "Task4-results-"+ (makeNominal ? "nominal" : "continuous") +"/"+classifierNames[c];
            expArgs.datasetName = "FiftyWords";
            expArgs.classifier = classifiers[c];

            if (new File("./results/"+expArgs.estimatorName+"/Predictions/"+expArgs.datasetName+"/testFold0.csv").exists()) {
                System.out.println("testFold0.csv already exists for " + classifierNames[c] + " on " + (makeNominal ? "nominal" : "continuous")+"/FiftyWords");
                continue;
            }

            ClassifierExperiments.buildExperimentDirectoriesAndFilenames(expArgs, classifiers[c]);
            long startTime = System.nanoTime();
            ClassifierExperiments.runExperiment(expArgs, new Instances(TRAIN), new Instances(TEST), classifiers[c]);
            long endTime = System.nanoTime();
            System.out.println("Done " + classifierNames[c] + ". took: " + ((endTime-startTime)/1000000000) + "s");
        }
        System.out.println("Testing done!");
    }

    public static void GetStats(String resultsPath, String taskName) throws Exception {
        File resPathFile = new File(resultsPath);
        String[] classifiers = new String[resPathFile.listFiles().length];
        int i = 0;
        for (File folder : resPathFile.listFiles()) {
            classifiers[i] = folder.getName();
            i++;
        }

        String[][] datasets = new String[][]{DatasetLists.nominalAttributeProblems, DatasetLists.continuousAttributeProblems};
        String[] datasetnames = new String[]{"nominal", "continuous"};

        for (int j = 0; j < datasets.length; j++) {
            new MultipleEstimatorEvaluation("./analysis", taskName + "-" + datasetnames[j], 1)
                    .setTestResultsOnly(true)
                    .setUseAllStatistics()
                    .setDatasets(datasets[j])
                    .readInEstimators(classifiers, classifiers, resultsPath)
                    .runComparison();
        }
    }

    public static void GetStatst8() throws Exception {
        File[] resPathFile = new File[]{
                new File("./results/Task4-results-nominal/"),
                new File("./results/Task4-results-continuous/")
        };

        String[] types = new String[]{"nominal", "continuous"};

        for (int j = 0; j < types.length; j++) {

            String[] classifiers = new String[resPathFile[j].listFiles().length];
            int i = 0;
            for (File folder : resPathFile[j].listFiles()) {
                classifiers[i] = folder.getName();
                i++;
            }


            new MultipleEstimatorEvaluation("./analysis", "FiftyWords" + "-" + types[j], 1)
                    .setTestResultsOnly(true)
                    .setUseAllStatistics()
                    .setDatasets(new String[]{"FiftyWords"})
                    .readInEstimators(classifiers, classifiers, resPathFile[j].getPath() + "/")
                    .runComparison();
        }
    }

    public static void makeCaseDiagrams() throws Exception {
        File[] folders = new File[]{new File("./analysis/FiftyWords-nominal"), new File("./analysis/FiftyWords-continuous")};
        File[] results = new File[]{new File("./results/Task4-results-nominal/"), new File("./results/Task4-results-continuous/")};


        File[] classifiers = results[0].listFiles();
        String[] classifier_names = new String[classifiers.length];
        for (int i = 0; i < classifiers.length; i++) {
            classifier_names[i] = classifiers[i].getName();
        }

        /** ROC Curve for Fifty words cont and nom*/
        String[] types = new String[]{"nominal", "continuous"};
        String[] datasets = new String[]{"FiftyWords"};

        int numFolds = 1;

        for (String type : types) {
            ClassifierResults[][] res = new ClassifierResults[classifier_names.length][numFolds];

            for (int i = 0; i < classifier_names.length;i++) {
                for (int f = 0; f < numFolds; f++) {
                    res[i][f] = new ClassifierResults("./results/Task4-results-"+type+"/" + classifier_names[i] + "/Predictions/" + datasets[0] + "/testFold" + f + ".csv");
                }
            }

            ClassifierResults[] concatResults = ClassifierResults.concatenateClassifierResults(res);

            ROCDiagramMaker.matlab_buildROCDiagrams(
                    "./analysis/FiftyWords-"+type+"/",
                    "FiftyWords-"+type,
                    datasets[0],
                    concatResults,
                    classifier_names,
                    false
            );
        }
    }

    public static void makeSummaryDiagramsTest(String taskName) throws Exception {
        String[][] classifier_names;
        String[] dataset_types = new String[]{"nominal", "continuous"};
        String target_results_folder;
        int numFolds = 1;
        if (Objects.equals(taskName, "Task1")) {
            //Id3 no continuous
            classifier_names = new String[][]{
                    new String[]{
                            "ChiTree",
                            "GiniTree",
                            "Id3",
                            "InformationGain",
                            "InformationGainRatio",
                            "J48"
                    },
                    new String[]{
                            "ChiTree",
                            "GiniTree",
                            "InformationGain",
                            "InformationGainRatio",
                            "J48"
                    }
            };
            target_results_folder = "./results/Task1-results/";
        } else if (Objects.equals(taskName, "Task2")) {
            classifier_names = new String[][]{
                    new String[]{
                            "CourseworkTreeEnsemble50",
                            "CourseworkTreeEnsemble50WDist",
                            "CourseworkTreeEnsemble100",
                            "CourseworkTreeEnsemble100WDist",
                            "CourseworkTreeTuned"
                    },
                    new String[]{}
            };

            target_results_folder = "./results/Task2-results/";
        } else if (Objects.equals(taskName, "Task3")) {
            classifier_names = new String[][]{
                    new String[]{
                            "RotationForrest",
                            "RandomForrest",
                            "AdaBoostM1",
                            "Vote",
                            "Stacking",
                            "IBk",
                            "J48",
                            "PART",
                            "NaiveBayes",
                            "OneR",
                            "SMO",
                            ////"Logistic",
                            "LogitBoost",
                            "TreeEnsemble"
                    },
                    new String[]{
                            "RotationForrest",
                            "RandomForrest",
                            "AdaBoostM1",
                            "Vote",
                            "Stacking",
                            "IBk",
                            "J48",
                            "PART",
                            "NaiveBayes",
                            "OneR",
                            "SMO",
                            ////"Logistic",
                            "LogitBoost",
                            "TreeEnsemble"
                    }
            };

            target_results_folder = "./results/Task3-results/";
        }
        else {
            return;
        }

        File[] dataset_folders = new File(target_results_folder).listFiles()[0].listFiles()[0].listFiles();
        String[][] datasets;
        if (dataset_folders.length > 22) {// we know its continuous and nominal
            datasets = new String[][] {DatasetLists.nominalAttributeProblems, DatasetLists.continuousAttributeProblems};
        } else {
            datasets = new String[2][dataset_folders.length];
            for (int i = 0; i < dataset_folders.length; i++) {
                datasets[0][i] = dataset_folders[i].getName();
            }
        }
        for (int cg = 0; cg < classifier_names.length; cg++) {
            if (datasets[cg].length == 0) { continue; }
            ClassifierResults[][][] res = new ClassifierResults[classifier_names[cg].length][datasets[cg].length][numFolds];

            for (int c = 0; c < classifier_names[cg].length; c++) {
                for (int d = 0; d < datasets[cg].length; d++) {
                    for (int f = 0; f < numFolds; f++) {
                        ClassifierResults resulting = new ClassifierResults(target_results_folder + classifier_names[cg][c] + "/Predictions/" + datasets[cg][d] + "/testFold" + f + ".csv");;
                        resulting.setDatasetName(taskName);
                        res[c][d][f] = resulting;
                        System.out.println("[" + datasets[cg][d] + "/" + classifier_names[cg][c] + "/0] = " + res[c].length + " | renamed to: " + res[c][d][f].getDatasetName());
                    }
                }
            }

            ClassifierResults[][] concatResults2D = new ClassifierResults[res.length][0];
            for (int d = 0; d < res.length; d++) {
                concatResults2D[d] = ClassifierResults.concatenateClassifierResults(res[d]);
                System.out.println("[" + datasets[cg][d] + "/0] = " + concatResults2D.length +"/"+ concatResults2D[d].length);
            }

            ClassifierResults[] concatenatedRes = ClassifierResults.concatenateClassifierResults(concatResults2D);

            for (int d = 0; d < concatenatedRes.length; d++) {
                System.out.println("[" + concatenatedRes[d].getDatasetName() + "] = " + concatenatedRes.length);
            }

            ROCDiagramMaker.matlab_buildROCDiagrams("./analysis/"+taskName+"-"+dataset_types[cg]+"/", taskName+"-collated", taskName,concatenatedRes, classifier_names[cg], false);
        }
    }

    public static void makeDiagrams(String taskName) throws Exception {
        File folder = new File("./analysis/"+taskName+"/");
        File resultFolder = new File("./results/"+taskName+"-results/");
        String[] classifierName = new String[resultFolder.listFiles().length];
        //String[] classifierNamebarid3 = new String[classifierName.length-1];
        //String[] datasets = new String[resultFolder.listFiles()[0].listFiles()[0].listFiles().length];
        String[][] datasets = new String[][]{DatasetLists.nominalAttributeProblems, DatasetLists.continuousAttributeProblems};
        String[] datasetnames = new String[]{"nominal", "continuous"};
        int numFolds = 1;

        /*int dsIndex = 0;
        for (File f : resultFolder.listFiles()[0].listFiles()[0].listFiles()){
            datasets[dsIndex] = f.getName();
            dsIndex++;
        }*/

        //System.out.println(Arrays.toString(datasets));

        int index = 0;
        for (File f : resultFolder.listFiles()) {
            classifierName[index] = f.getName();
            index++;
        }

        /*int nameIndex = 0;
        for (String name : classifierName) {
            if (Objects.equals(name, "Id3")) { continue; }
            //System.out.println(name);
            classifierNamebarid3[nameIndex] = name;
            nameIndex++;
        }*/

        System.out.println(Arrays.toString(classifierName));

        int dsType = 0;
        for (String[] datasetType : datasets) {
            System.out.println(datasetnames[dsType]);
            for (String selDataset : datasetType) {
                if (new File("./analysis/" + taskName + (datasets.length == 2 ? "-" + datasetnames[dsType] : "") + "/dias_ROCCurve/" + "rocDia_"+taskName + (datasets.length == 2 ? "-" + datasetnames[dsType] : "") +"_"+selDataset+".pdf").exists()) {
                    System.out.println("["+taskName + "-" + datasetnames[dsType]+"/"+selDataset+"] already exists!");
                    continue;
                }
                ClassifierResults[][] res = new ClassifierResults[classifierName.length][numFolds];
                int res_slot = 0;
                for (int i = 0; i < classifierName.length;i++) {
                    //if (dsType == 1 && Objects.equals(classifierName[i], "Id3")) {
                    //    System.out.println("skipped: " + classifierName[i]);continue; }
                    for (int f = 0; f < numFolds; f++) {
                        System.out.println("./results/" + taskName + "-results/" + classifierName[i] + "/Predictions/" + selDataset + "/testFold" + f + ".csv");
                        res[res_slot][f] = new ClassifierResults("./results/" + taskName + "-results/" + classifierName[i] + "/Predictions/" + selDataset + "/testFold" + f + ".csv");
                    }
                    res_slot++;
                }

                ClassifierResults[] concatenatedRes = ClassifierResults.concatenateClassifierResults(res);
                System.out.println(res.length);
                for (ClassifierResults cr : concatenatedRes) {
                    System.out.println(cr.getClassifierName());
                }
                ROCDiagramMaker.matlab_buildROCDiagrams("./analysis/" + taskName + (datasets.length == 2 ? "-" + datasetnames[dsType] : "") + "/", taskName + (datasets.length == 2 ? "-" + datasetnames[dsType] : "") , selDataset, concatenatedRes, classifierName, false);
            }
            dsType++;
        }
    }

    public static void makeCriticalDifferenceDiagrams(String taskName) throws IOException {
        File[] taskAnalysis = new File[]{
                new File("analysis/"+taskName+"-nominal"),
                new File("analysis/"+taskName+"-continuous")
        };

        int taskid = 0;
        for (File task_directory : taskAnalysis) {
            if (!task_directory.exists()) { continue;}
            File critDiff = new File(task_directory.getAbsolutePath() + "/dias_CriticalDifference/");
            for (File critDiffType : critDiff.listFiles()) {
                if (critDiffType.getAbsolutePath().endsWith(".txt")) { continue; }
                if (!critDiffType.getName().equals("friedman")) { continue; }
                for (File csv : critDiffType.listFiles()) {
                    //System.out.println(!csv.getAbsolutePath().endsWith(".csv") + " - " + critDiffType.getAbsolutePath());
                    if (!csv.getAbsolutePath().endsWith(".csv")) { continue;}
                    //System.out.println(csv.getName().substring(taskid == 0 ? 17:20, csv.getName().length()-4));
                    createCritDiffDiag(csv, csv.getName().substring(csv.getName().indexOf("_")+1, csv.getName().length()-4));
                }
            }
            taskid++;
        }
        MatlabController.getInstance().discconnectMatlab();
    }

    public static void createCritDiffDiag(File file, String taskname) throws FileNotFoundException {
        MatlabController proxy = MatlabController.getInstance();
        proxy.eval("cd C:\\Users\\Grimby\\Documents\\GitHub\\tsml\\src\\main\\matlab");
        Scanner reader = new Scanner(file);
        //Scanner scanner = new Scanner(System.in);
        String labels = "[\""+reader.nextLine().replaceAll(",", "\",\"")+"\"]";
        StringBuilder data = new StringBuilder();
        data.append("[");
        int totalLines = 0;
        String line = "";
        while (reader.hasNext()) {
            data.append("[");
            line = reader.nextLine();
            data.append(line).append("],").append("\n");
            totalLines++;
        }
        if (totalLines == 1) {
            data.append("[");
            data.append(line).append("],").append("\n");
        }
        data.append("]");
        System.out.println("critdiff(\""+taskname+"\","+data.toString()+", "+labels+")");
        proxy.eval("critdiff(\""+taskname+"\","+data.toString()+", "+labels+")");
        //System.out.println("Graph title: " + taskname);
        //scanner.nextLine();
    }
}
