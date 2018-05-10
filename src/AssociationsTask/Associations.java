package AssociationsTask;

import weka.associations.Apriori;
import weka.associations.FPGrowth;
import weka.core.Instances;

public class Associations {
    //Apriori
    public static void apriori(Instances data) throws Exception {
        Apriori model = new Apriori();
        model.buildAssociations(data);
        System.out.println(model);
    }
    //FPGrowth
    public static void fpGrowth(Instances data) throws Exception{
        FPGrowth model = new FPGrowth();
        model.buildAssociations(data);
        System.out.println(model);
    }
}
