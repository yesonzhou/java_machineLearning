package RegressionTask;


import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

import javax.swing.*;
import java.util.Random;

public class Regression {
    // 线性回归
    public static void linearRegression(Instances data) throws Exception {
        LinearRegression model = new LinearRegression();
        model.buildClassifier(data);
        System.out.println(model);

        //获取系数
        double coef[] = model.coefficients();
        System.out.println(coef);

        evaluateClassifier(model,data);
    }

    // 回归树
    public static void  mp5Tree(Instances data) throws Exception {
        M5P model = new M5P();
        model.setOptions(new String[] { "" });
        model.buildClassifier(data);
        System.out.println(model);
        evaluateClassifier(model,data);
    }


    // 评价分类器
    public static void evaluateClassifier(Classifier classifier, Instances data) throws Exception {
        /* 五、评估与预测误差度量
		 * Evaluation
		 */

        Evaluation eval_roc = new Evaluation(data);
        eval_roc.crossValidateModel(classifier, data, 10, new Random(1), new Object[] {}); //这里用了交叉验证
        System.out.println(eval_roc.toSummaryString());
        // Confusion matrix 混淆矩阵（通过产生的混淆矩阵可以进一步查看错误分类出现在什么地方）
//        double[][] confusionMatrix = eval_roc.confusionMatrix();
//        System.out.println(eval_roc.toMatrixString());

		/* ROC曲线
		 * Bonus: Plot ROC curve
		 */

//        ThresholdCurve tc = new ThresholdCurve();
//        int classIndex = 0;
//        Instances result = tc.getCurve(eval_roc.predictions(), classIndex);
//        // plot curve
//        ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
//        vmc.setROCString("(Area under ROC = " + tc.getROCArea(result) + ")");
//        vmc.setName(result.relationName());
//        PlotData2D tempd = new PlotData2D(result);
//        tempd.setPlotName(result.relationName());
//        tempd.addInstanceNumberAttribute();
//        // specify which points are connected
//        boolean[] cp = new boolean[result.numInstances()];
//        for (int n = 1; n < cp.length; n++)
//            cp[n] = true;
//        tempd.setConnectPoints(cp);
//
//        // add plot
//        vmc.addPlot(tempd);
//        // display curve
//        JFrame frameRoc = new javax.swing.JFrame("ROC Curve");
//        frameRoc.setSize(800, 500);
//        frameRoc.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        frameRoc.getContentPane().add(vmc);
//        frameRoc.setVisible(true);
    }

}
