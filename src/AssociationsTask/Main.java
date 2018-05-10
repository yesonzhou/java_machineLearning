package AssociationsTask;

import weka.associations.Apriori;
import weka.associations.FPGrowth;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;

import static AssociationsTask.Associations.apriori;
import static AssociationsTask.Associations.fpGrowth;

public class Main {
    public static void main(String args[]) throws Exception {
        // load data
        Instances data = new Instances(new BufferedReader(new FileReader("data/associations/supermarket.arff")));
        // build model
        apriori(data);
        fpGrowth(data);

    }
}
