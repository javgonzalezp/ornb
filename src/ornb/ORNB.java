package ornb;

import java.util.Enumeration;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class ORNB{
  
  /**
   * Main method for this class.
   *
   * @param argv the options
 * @throws Exception 
   */
  public static void main(String[] argv) throws Exception {
		DataSource loader;
		Instances data;
		
		loader = new DataSource("iris.arff");
		data = loader.getDataSet();
		Forest f = new Forest(10);
   		//Predictions(nb, data);

		f.initializeForest();
		
		//entrenamiento con iris
	    Enumeration enu = data.enumerateInstances();
	    //for(int j=0; j<100; j++){
	    	//enu = data.enumerateInstances();
		    while (enu.hasMoreElements()) {
		    	 Instance instance = (Instance) enu.nextElement();
		    	 f.addElement(instance.toString());
		    }
	    //}
	    //testing con iris
	    enu = data.enumerateInstances();
	    while (enu.hasMoreElements()) {
	    	 Instance instance = (Instance) enu.nextElement();
	    	 double[] a = f.classify(instance.toString());
	    	 System.out.println(Utils.maxIndex(a));
	    }
	    
	//    */
/*
	  ORNB ornb = new ORNB();
	    try {
	    	DataSource loader;
	    	Instances data;
	    	SparseInstance i=new SparseInstance(3);
	    	
			loader = new DataSource("iris.arff");
			data = loader.getDataSet();
			
	   		if (data.classIndex() == -1)
	   			   data.setClassIndex(data.numAttributes()-1);
	   		i.setDataset(data);

	   	    Resample rs = new Resample();
	   	    Random r = new Random();
   			double a = r.nextDouble();
	   	    
   			rs.setSampleSizePercent(a*100);
	   	    rs.setInputFormat(data);
	   	    rs.setRandomSeed(10);
	   	    
	   	    ornb.setPrintTrees(true);
	   	    ornb.buildClassifier(data);
	   		data = Resample.useFilter(data, rs);
	   		ornb.Predictions(ornb, data);
	   		System.out.println(data.toString());
	   		
			for(int j=0; j<lista.size(); j++)
				System.out.println(Utils.maxIndex(lista.get(j)));
	      }
	      catch (Exception e) {
	        if (    ((e.getMessage() != null) && (e.getMessage().indexOf("General options") == -1))
	            || (e.getMessage() == null) )
	          e.printStackTrace();
	        else
	          System.err.println(e.getMessage());
	      }
	      Histogram h = new Histogram(5);
	      h.updateHistogram(23.0);
	      h.updateHistogram(19);
	      h.updateHistogram(10);
	      h.updateHistogram(16);
	      h.updateHistogram(36);
	      h.updateHistogram(2);
	      h.updateHistogram(9);
	      h.updateHistogram(32);
	      h.updateHistogram(30);
	      h.updateHistogram(45);
	      
	      h.printHistogram();
	      double a=h.getMean();
	      double b=h.getStandarDeviation();
	      System.out.println("Mean: "+h.getMean()+" Standar Deviation: "+h.getStandarDeviation());
	      System.out.println("Puntos menores que 15: " + h.sumProcedure(15));
	      */
  }
}

