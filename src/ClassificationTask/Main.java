package ClassificationTask;


import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import static ClassificationTask.classifiers.*;

public class Main {
    public static void main(String[] args) throws Exception {

        /* 一、加载数据
		 * Load the data
		 * 创建DataSource对象，用户接收各种文件格式，并将其转换成Instances
		 */
        DataSource source = new DataSource("data/classification/zoo.arff");
        Instances data = source.getDataSet();
        System.out.println(data.numInstances() + " instances loaded.");
        // System.out.println(data.toString());

        // remove animal attribute 移走第一个属性，其余属性用作数据集，用于训练分类器
        String[] opts = new String[] { "-R", "1" };
        Remove remove = new Remove();
        remove.setOptions(opts);
        remove.setInputFormat(data);
        data = Filter.useFilter(data, remove);

		/*	二、 特征选择
		 * Feature selection
		 * weka中提供AttributeSelection对象进行属性选择
		 * 它需要两个额外的参数：评价器（evaluator），排行器（ranker）
		 */
//        AttributeSelection attSelect = new AttributeSelection();
//        InfoGainAttributeEval eval = new InfoGainAttributeEval();	//这里选择信息增益用作评价器，然后通过信息增益分数对特征进行排序
//        Ranker search = new Ranker();
//        attSelect.setEvaluator(eval);
//        attSelect.setSearch(search);
//        attSelect.SelectAttributes(data);
//        int[] indices = attSelect.selectedAttributes();
//        System.out.println("Selected attributes: "+Utils.arrayToString(indices));

		/* 三、学习算法
		 * Build a decision tree
		 */
//        J48Tree(data);
//        knnIBK(data);
//        naiveBayes(data);
//        multilayerPerceptron(data);
//        adaboostM1(data);
        smo(data);

    }
}
