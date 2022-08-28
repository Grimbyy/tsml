package ml_6002b_coursework;

import experiments.ClassifierExperiments;
import experiments.ExperimentalArguments;
import experiments.data.DatasetLoading;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ThreadedEvaluation {

    class Experiment implements Runnable{

        Classifier[] classifiers;
        String[] classifier_names;
        String dataset_location;
        String dataset;
        String taskName;

        public Experiment(String task_name, Classifier[] classifiers, String[] classifier_names, String dataset_location, String dataset) {
            this.classifiers = classifiers;
            this.classifier_names = classifier_names;
            this.dataset_location = dataset_location;
            this.dataset = dataset;
            this.taskName = task_name;
        }

        @Override
        public void run() {
            Instances data = DatasetLoading.loadDataNullable(dataset_location+"/"+dataset+"/"+dataset+".arff");
            Instances[] splitData = InstanceTools.resampleInstances(data, seed, 0.5);

            for (int c = 0; c < classifiers.length; c++) {

                if (new File(result_folder+"/"+taskName+"/"+classifier_names[c]+"/Predictions/"+dataset+"/testFold0.csv").exists()) {
                    System.out.println("[Skipped/" + classifier_names[c] + "/" + dataset + "]");
                    continue;
                }

                ExperimentalArguments expArgs = new ExperimentalArguments();
                expArgs.datasetName = dataset;
                expArgs.estimatorName = classifier_names[c];
                expArgs.classifier = classifiers[c];

                ClassifierExperiments.runExperiment(expArgs, new Instances(splitData[0]), new Instances(splitData[1]), classifiers[c]);
                System.out.println("[Complete/"+dataset+"/"+classifier_names[c]+"]");
            }

            System.out.println("[Complete/"+dataset+"] Finished Experiments");
        }
    }

    static String result_folder = "./results";
    static long seed = 0;

    public static void buildExperimentDirectories(String task_name, String[] classifiers, String[] datasets) {
        File resFolder = new File(result_folder);
        if (!resFolder.exists()) {
            System.err.println("[ERROR] Failed to find directory '" + result_folder + "'. Creating one...");
            resFolder.mkdirs();
        }

        for (int c = 0; c < classifiers.length; c++) {
            for (int d = 0; d < datasets.length; d++) {
                File resLocation = new File(result_folder + "/" + task_name + "/" + classifiers[c] + "/Predictions/" + datasets[d] + "/");
                File wrkLocation = new File(result_folder + "/" + task_name + "/" + classifiers[c] + "/Workspace/" + datasets[d] + "/");
                if (!resLocation.mkdirs() || !wrkLocation.mkdirs()) {
                    System.err.println("Failed to make folder '" + result_folder + "/" + classifiers[c] + "/");
                }
            }
        }
    }

    public void runExperimentsThreaded(String task_name, Classifier[] classifiers, String[] classifier_names, String dataset_location, String[] datasets) {

        int numCores = Runtime.getRuntime().availableProcessors();
        int numThreads = datasets.length;

        System.out.println("/====== Threading ======/");
        System.out.println("/\tCores: "+numCores);
        System.out.println("/\tThreads: "+numThreads);

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        buildExperimentDirectories(task_name, classifier_names, datasets);
        for (int i = 0; i < datasets.length; i++) {
            executor.execute(new ThreadedEvaluation.Experiment(task_name, classifiers, classifier_names, dataset_location, datasets[i]));
        }

        executor.shutdown();
        while (!executor.isTerminated()) {

        }
        System.out.println("[Complete/All Threads]");
    }
}
