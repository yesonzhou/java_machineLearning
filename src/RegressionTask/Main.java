package RegressionTask;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.util.Random;

import static RegressionTask.Regression.linearRegression;
import static RegressionTask.Regression.mp5Tree;

public class Main {
    public static void main(String[] args) throws Exception {
        /*
		 * Load data
		 */
        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator(",");
        loader.setSource(new File("data/regression/ENB2012_data.csv"));
        Instances data = loader.getDataSet();

        // System.out.println(data);

		/*
		 * Build regression models
		 */
        // set class index to Y1 (heating load)
        data.setClassIndex(data.numAttributes() - 2);
        // remove last attribute Y2
        Remove remove = new Remove();
        remove.setOptions(new String[] { "-R", data.numAttributes() + "" });
        remove.setInputFormat(data);
        data = Filter.useFilter(data, remove);

        // build a regression model 建一个线性回归模型
        linearRegression(data);

        // build a regression tree model	回归树模型
        mp5Tree(data);


		/*
		 * Bonus: Build additional models
		 */
        // ZeroR modelZero = new ZeroR();
        //
        //
        //
        //
        //
//         REPTree modelTree = new REPTree();
        // modelTree.buildClassifier(data);
        // System.out.println(modelTree);
        // eval = new Evaluation(data);
        // eval.crossValidateModel(modelTree, data, 10, new Random(1), new
        // String[]{});
        // System.out.println(eval.toSummaryString());
        //
        // SMOreg modelSVM = new SMOreg();
        //
        // MultilayerPerceptron modelPerc = new MultilayerPerceptron();
        //
        // GaussianProcesses modelGP = new GaussianProcesses();
        // modelGP.buildClassifier(data);
        // System.out.println(modelGP);
        // eval = new Evaluation(data);
        // eval.crossValidateModel(modelGP, data, 10, new Random(1), new
        // String[]{});
        // System.out.println(eval.toSummaryString());

		/*
		 * Bonus: Save ARFF
		 */
        // ArffSaver saver = new ArffSaver();
        // saver.setInstances(data);
        // saver.setFile(new File(args[1]));
        // saver.setDestination(new File(args[1]));
        // saver.writeBatch();
    }
}
