package ClassificationTask;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

import javax.swing.*;
import java.util.Random;

public class classifiers {

    //决策树：J48是weka中在基于C4.5是实现的决策树算法
    public static void J48Tree(Instances data) throws Exception {
        // 初始化决策树输入参数
        String[] options = new String[]{"-U"};
        J48 model = new J48();
        model.setOptions(options);
        model.buildClassifier(data);
        System.out.println(model);

        /**
         * 可以进行单例测试，也可以对训练集测试
         */
        //运用测试集
//        Instances trainData = new Instances(data,0,95);
//        Instances testData = new Instances(data,95,data.numInstances()-95);
//        model.buildClassifier(trainData);
//        Evaluation eval = new Evaluation(trainData);
//        eval.evaluateModel(model,testData);
//        System.out.println("测试集测试："+ eval.toSummaryString());

        //单例测试
//        double[] vals = new double[data.numAttributes()];
//        vals[0] = 1.0; // hair {false, true}
//        vals[1] = 0.0; // feathers {false, true}
//        vals[2] = 0.0; // eggs {false, true}
//        vals[3] = 1.0; // milk {false, true}
//        vals[4] = 0.0; // airborne {false, true}
//        vals[5] = 0.0; // aquatic {false, true}
//        vals[6] = 0.0; // predator {false, true}
//        vals[7] = 1.0; // toothed {false, true}
//        vals[8] = 1.0; // backbone {false, true}
//        vals[9] = 1.0; // breathes {false, true}
//        vals[10] = 1.0; // venomous {false, true}
//        vals[11] = 0.0; // fins {false, true}
//        vals[12] = 4.0; // legs INTEGER [0,9]
//        vals[13] = 1.0; // tail {false, true}
//        vals[14] = 1.0; // domestic {false, true}
//        vals[15] = 0.0; // catsize {false, true}
//        Instance myUnicorn = new DenseInstance(1.0, vals);
//        //Assosiate your instance with Instance object in this case dataRaw
//        myUnicorn.setDataset(data);
//
//        double label = model.classifyInstance(myUnicorn);
//        System.out.println("预测的动物类型是："+String.valueOf(data.classAttribute().value((int) label)));

        // Visualize decision tree
//        TreeVisualizer tv = new TreeVisualizer(null, tree.graph(),
//                new PlaceNode2());
//        JFrame frame = new javax.swing.JFrame("Tree Visualizer");
//        frame.setSize(800, 500);
//        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        frame.getContentPane().add(tv);
//        frame.setVisible(true);
//        tv.fitToScreen();
        /* 五、评估与预测误差度量*/
        Classifier j48 = new J48();
        evaluateClassifier(j48,data);
    }

    //knn：weka中实现knn的算法有IB1、IBk，即取1和k个邻近邻居
    public  static void knnIBK(Instances data) throws Exception {
        IBk model = new IBk();
        model.setKNN(3);
        model.buildClassifier(data);
        System.out.println(model);

        evaluateClassifier(new IBk(),data);
    }

    //贝叶斯算法
    public static void naiveBayes(Instances data) throws Exception {
        NaiveBayes model = new NaiveBayes();
        model.buildClassifier(data);
        System.out.println(model);
        evaluateClassifier(new NaiveBayes(),data);
    }

    //支持向量机
    public static void smo(Instances data) throws Exception {
        SMO model = new SMO();
        model.buildClassifier(data);
        System.out.println(model);
        evaluateClassifier(model,data);
    }

    //神经网络
    public static void multilayerPerceptron(Instances data) throws Exception {
        MultilayerPerceptron model = new MultilayerPerceptron();
        model.buildClassifier(data);
        System.out.println(model);
        evaluateClassifier(new MultilayerPerceptron(),data);
    }

    //Adaboost
    public static void adaboostM1(Instances data) throws Exception {
        AdaBoostM1 model = new AdaBoostM1();
        model.buildClassifier(data);
        System.out.println(model);
        evaluateClassifier(new AdaBoostM1(),data);
    }




    // 评价分类器
    public static void evaluateClassifier(Classifier classifier,Instances data) throws Exception {
        /* 五、评估与预测误差度量
		 * Evaluation
		 */

        Evaluation eval_roc = new Evaluation(data);
        eval_roc.crossValidateModel(classifier, data, 10, new Random(1), new Object[] {}); //这里用了交叉验证
        System.out.println(eval_roc.toSummaryString());
        // Confusion matrix 混淆矩阵（通过产生的混淆矩阵可以进一步查看错误分类出现在什么地方）
//        double[][] confusionMatrix = eval_roc.confusionMatrix();
        System.out.println(eval_roc.toMatrixString());

		/* ROC曲线
		 * Bonus: Plot ROC curve
		 */

//        ThresholdCurve tc = new ThresholdCurve();
//        int classIndex = 0;
//        Instances result = tc.getCurve(eval_roc.predictions(), classIndex);
//        // plot  curve
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
