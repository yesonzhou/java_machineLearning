package ClusteringTask;


import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;

import static ClusteringTask.Clustering.*;

public class Main {
    public static void main(String args[]) throws Exception{

        //load data
        Instances data = new Instances(new BufferedReader(new FileReader("data/clustering/bank-data.arff")));

        //k-meank
        kmeans(data);

        // EM（Expectation Maximization,期望最大化）
//        em(data);

        // DBSCAN（Density-based spatial clustering of applications with noise）基于密度
//        dbscan(data);

        // HierachicalClusterer 层次聚类
//        hierachicalClusterer(data);

        // Affinity propagation Clustering Algorithm 消息传递算法


    }
}
