package tsml.classifiers.distance_based.utils.experiment;

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.classifiers.Copier;
import tsml.classifiers.distance_based.utils.strings.StrUtils;

import java.sql.Time;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.logging.Level;
import java.util.stream.Collectors;

public class ExperimentConfig implements Copier {
    
    public ExperimentConfig() {}

    @Parameter(names = {"-c", "--classifier"}, description = "The classifier to use.")
    private String classifierName;

    @Parameter(names = {"-s", "--seed"}, description = "The seed used in resampling the data and producing random numbers in the classifier.")
    private Integer seed;

    @Parameter(names = {"-r", "--results"}, description = "The results directory.")
    private String resultsDirPath;

    @Parameter(names = {"-d", "--data"}, description = "The data directory.")
    private String dataDirPath;

    @Parameter(names = {"-p", "--problem"}, description = "The problem name. E.g. GunPoint.")
    private String problemName;

    @Parameter(names = {"--cp", "--checkpoint"}, description = "Periodically save the classifier to disk. Default: off")
    private boolean checkpoint = false;

    @Parameter(names = {"--rcp", "--removeCheckpoint"}, description = "Remove any checkpoints upon completion of a train time contract. The assumption here is that once the classifier has built in the given time limit, no further work will be done and the checkpoint can be safely removed. In other words, the assumption is that the checkpoint is only useful if the classifier gets stoppped mid-build and must be restarted. When the classifier finishes building, the checkpoint files are redundant, therefore. Note this does not affect multiple contracts as the checkpoint files are copied before removal. I.e. a contract of 1h completes and leaves behind some checkpoint files. These are copied over to the subsequent 2h contract before removal from the 1h contract working area. Default: off")
    private boolean removeCheckpoint = false;

    @Parameter(names = {"--cpi", "--checkpointInterval"}, description = "The minimum interval between checkpoints. Classifiers may not produce checkpoints within the checkpoint interval time from the last checkpoint. Therefore, classifiers may produce checkpoints with (much) larger intervals inbetween, depending on their checkpoint frequency. Default: 1h")
    private String checkpointInterval = "1h";

    @Parameter(names = {"--scp", "--snapshotCheckpoints"}, description = "Keep every checkpoint as a snapshot of the classifier model at that point in time. When disabled, classifiers overwrite their last checkpoint. When enabled, classifiers will write checkpoints with a time stamp rather than overwriting previous checkpoints. Default: off")
    private boolean snapshotCheckpoints = false;

    @Parameter(names = {"--ttl", "--trainTimeLimit"}, description = "Contract the classifier to build in a set time period. Give this option two arguments in the form of '--contractTrain <amount> <units>', e.g. '--contractTrain 5 minutes'")
    private List<String> trainTimeLimitStrs = new ArrayList<>();
    private List<TimeSpan> trainTimeLimits = new ArrayList<>();

    @Parameter(names = {"-e", "--evaluate"}, description = "Estimate the train error. Default: false")
    private boolean evaluateClassifier = false;

    @Parameter(names = {"-t", "--threads"}, description = "The number of threads to use. Set to 0 or less for all available processors at runtime. Default: 1")
    private int numThreads = 1;

    @Parameter(names = {"--trainOnly"}, description = "Only train the classifier, do not test.")
    private boolean trainOnly = false;

    @Parameter(names = {"-o", "--overwrite"}, description = "Overwrite previous results.")
    private boolean overwriteResults = false;

    public TimeSpan getTrainTimeLimit() {
        return trainTimeLimit;
    }

    public void setTrainTimeLimit(final TimeSpan trainTimeLimit) {
        this.trainTimeLimit = trainTimeLimit;
    }

    private static class LogLevelConverter implements IStringConverter<Level> {

        @Override public Level convert(final String s) {
            return Level.parse(s.toUpperCase());
        }
    }

    @Parameter(names = {"-l", "--logLevel"}, description = "The amount of logging. This should be set to a Java log level. Default: OFF", converter = LogLevelConverter.class)
    private Level logLevel = Level.OFF;

    private TimeSpan trainTimeLimit; // the current train time limit
    
    public ExperimentConfig(String[] args) {
        JCommander.newBuilder()
                .addObject(this)
                .build()
                .parse(args);
        // check the state of this experiment is set up
        if(seed == null) throw new IllegalStateException("seed not set");
        if(classifierName == null) throw new IllegalStateException("classifier name not set");
        if(problemName == null) throw new IllegalStateException("problem name not set");
        if(dataDirPath == null) throw new IllegalStateException("data dir path not set");
        if(resultsDirPath == null) throw new IllegalStateException("results dir path not set");
        // default to no train time contracts
        trainTimeLimits = new ArrayList<>();
        for(String trainTimeLimitStr : trainTimeLimitStrs) {
            trainTimeLimits.add(new TimeSpan(trainTimeLimitStr));
        }
        if(trainTimeLimits.isEmpty()) {
            // add a null limit to indicate there is no limit
            trainTimeLimits.add(null);
        } else {
            // sort the train contracts in asc order
            trainTimeLimits = trainTimeLimits.stream().distinct().sorted().collect(Collectors.toList());
        }
        // default to all cpus
        if(numThreads < 1) {
            numThreads = Runtime.getRuntime().availableProcessors();
        }
        Assert.assertFalse(trainTimeLimits.isEmpty());
    }

    public String getClassifierName() {
        return classifierName;
    }

    public Integer getSeed() {
        return seed;
    }

    public String getResultsDirPath() {
        return resultsDirPath;
    }

    public String getDataDirPath() {
        return dataDirPath;
    }

    public String getProblemName() {
        return problemName;
    }

    public boolean isCheckpoint() {
        return checkpoint;
    }

    public boolean isRemoveCheckpoint() {
        return removeCheckpoint;
    }

    public String getCheckpointInterval() {
        return checkpointInterval;
    }

    public boolean isSnapshotCheckpoints() {
        return snapshotCheckpoints;
    }

    public List<String> getTrainTimeLimitStrs() {
        return trainTimeLimitStrs;
    }

    public List<TimeSpan> getTrainTimeLimits() {
        return trainTimeLimits;
    }

    public boolean isEvaluateClassifier() {
        return evaluateClassifier;
    }

    public int getNumThreads() {
        return numThreads;
    }

    public boolean isTrainOnly() {
        return trainOnly;
    }

    public boolean isOverwriteResults() {
        return overwriteResults;
    }

    public Level getLogLevel() {
        return logLevel;
    }

    public void setClassifierName(final String classifierName) {
        this.classifierName = classifierName;
    }

    public void setSeed(final Integer seed) {
        this.seed = seed;
    }

    public void setResultsDirPath(final String resultsDirPath) {
        this.resultsDirPath = resultsDirPath;
    }

    public void setDataDirPath(final String dataDirPath) {
        this.dataDirPath = dataDirPath;
    }

    public void setProblemName(final String problemName) {
        this.problemName = problemName;
    }

    public void setCheckpoint(final boolean checkpoint) {
        this.checkpoint = checkpoint;
    }

    public void setRemoveCheckpoint(final boolean removeCheckpoint) {
        this.removeCheckpoint = removeCheckpoint;
    }

    public void setCheckpointInterval(final String checkpointInterval) {
        this.checkpointInterval = checkpointInterval;
    }

    public void setSnapshotCheckpoints(final boolean snapshotCheckpoints) {
        this.snapshotCheckpoints = snapshotCheckpoints;
    }

    public void setTrainTimeLimitStrs(final List<String> trainTimeLimitStrs) {
        this.trainTimeLimitStrs = Objects.requireNonNull(trainTimeLimitStrs);
    }

    public void setTrainTimeLimits(final List<TimeSpan> trainTimeLimits) {
        this.trainTimeLimits = trainTimeLimits;
    }

    public void setEvaluateClassifier(final boolean evaluateClassifier) {
        this.evaluateClassifier = evaluateClassifier;
    }

    public void setNumThreads(final int numThreads) {
        this.numThreads = numThreads;
    }

    public void setTrainOnly(final boolean trainOnly) {
        this.trainOnly = trainOnly;
    }

    public void setOverwriteResults(final boolean overwriteResults) {
        this.overwriteResults = overwriteResults;
    }

    public void setLogLevel(final Level logLevel) {
        this.logLevel = Objects.requireNonNull(logLevel);
    }


    public String getTestFilePath() {
        return StrUtils.joinPath(getExperimentResultsDirPath(), "testFold" + seed + ".csv");
    }

    public String getTrainFilePath() {
        return StrUtils.joinPath(getExperimentResultsDirPath(), "trainFold" + seed + ".csv");
    }

    public String getClassifierNameInResults() {
        String classifierNameInResults = classifierName;
        if(trainTimeLimit != null) {
            classifierNameInResults += "_" + trainTimeLimit;
        }
        return classifierNameInResults;
    }

    public String getExperimentResultsDirPath() {
        return StrUtils.joinPath( getResultsDirPath(), getClassifierNameInResults(), "Predictions", getProblemName());
    }

    public String getLockFilePath() {
        return StrUtils.joinPath(getExperimentResultsDirPath(), "fold" + getSeed());
    }

    public String getCheckpointDirPath() {
        return StrUtils.joinPath( getResultsDirPath(), getClassifierNameInResults(), "Workspace", getProblemName(), "fold" + getSeed());
    }
}
