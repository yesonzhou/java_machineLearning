package ClusteringTask;

import weka.clusterers.*;
import weka.core.Instances;

import java.util.Random;

public class Clustering {
    // simpleKmeans
    public static void kmeans(Instances data) throws Exception {
        SimpleKMeans model = new SimpleKMeans();
        model.buildClusterer(data);
        System.out.println(model);
        evaluateClusterer(model,data);
    }

    // EM（Expectation Maximization,期望最大化）
    public static void em(Instances data) throws Exception {
        EM model = new EM();
        model.buildClusterer(data);
        System.out.println(model);

        //使用对数似然度量评估聚类算法的质量（测量被识别的簇的一致程度）
//        double logLikelihood = ClusterEvaluation.crossValidateModel(model, data, 10, new Random(1));
//        System.out.println(logLikelihood);
        evaluateClusterer(model,data);
    }

    // DBSCAN（Density-based spatial clustering of applications with noise）基于密度
    public static void dbscan(Instances data) throws Exception {
        MakeDensityBasedClusterer model = new MakeDensityBasedClusterer();
        model.buildClusterer(data);
        System.out.println(model);
        evaluateClusterer(model,data);
    }

    // HierachicalClusterer 层次聚类
    public static void hierachicalClusterer(Instances data) throws Exception {
        HierarchicalClusterer model = new HierarchicalClusterer();
        model.buildClusterer(data);
        System.out.println(model);
        evaluateClusterer(model,data);
    }

    // Affinity propagation Clustering Algorithm 消息传递算法


    public static void evaluateClusterer(Clusterer model, Instances data) throws Exception {
        ClusterEvaluation evals = new ClusterEvaluation();
        evals.setClusterer(model);
        evals.evaluateClusterer(data);

        // l 输出评价结果
        System.out.println(evals.clusterResultsToString());

    }
}
