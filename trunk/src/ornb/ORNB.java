package ornb;

import java.util.Enumeration;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
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
		
		//NaiveBayes nb = new NaiveBayes();
		
		if (data.classIndex() == -1)
			   data.setClassIndex(data.numAttributes()-1);
		
		//nb.buildClassifier(data);
		  //Enumeration enu2 = data.enumerateInstances();
		    //for(int j=0; j<100; j++){
		    	//enu = data.enumerateInstances();
			//    while (enu2.hasMoreElements()) {
			  //  	 Instance instance = (Instance) enu2.nextElement();
			    //	 double[] aux = nb.distributionForInstance(instance);
			    //}
		
		
		//obtain the attributes of the arff
		String[] attributes = new String [data.numAttributes()-1];
		for(int i=0; i<data.numAttributes()-1; i++)
			attributes[i]=data.attribute(i).name();

		//obtain the classes of the arff
		String[] classes = new String [data.numClasses()];
		for(int i=0; i<data.numClasses(); i++)
			classes[i]=data.attribute(data.numAttributes()-1).value(i);
		
		Forest f = new Forest(10, classes, attributes);
   		//Predictions(nb, data);

		f.initializeForest();
		
		//entrenamiento con iris
	    Enumeration enu = data.enumerateInstances();
	    //for(int j=0; j<100; j++){
	    	//enu = data.enumerateInstances();
		    while (enu.hasMoreElements()) {
		    	 Instance instance = (Instance) enu.nextElement();
		    	 f.addElement(instance.toString(), instance.stringValue(instance.numValues()-1));
		    }
	    //}
		
		int[][] matrix = initializeConfusionMatrix(data.numClasses());
	    //testing con iris
		double acc = 0.0;
	    enu = data.enumerateInstances();
	    while (enu.hasMoreElements()) {
	    	 Instance instance = (Instance) enu.nextElement();
	    	 double[] a = f.classify(instance.toString());
	    	 if(instance.classValue()==(double)Utils.maxIndex(a))
	    		 acc++;
	    	 matrix[Utils.maxIndex(a)][(int) instance.classValue()]++;
	    	 //System.out.println(instance.classValue()+" : "+(double)Utils.maxIndex(a));
	    	 //data.
	    }
	    System.out.println("Accurracy: "+acc/data.numInstances());
	    printConfusionMatrix(matrix, data.numClasses());
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

private static void printConfusionMatrix(int[][] matrix, int numClasses) {
	System.out.println("Confusion Matrix");
	for(int j=0; j<numClasses; j++){
		for(int k=0; k<numClasses; k++)
			System.out.print(matrix[j][k]+" ");
		System.out.println();
	}
}

private static int[][] initializeConfusionMatrix(int i) {
	int[][] matrix = new int[i][i];
	
	for(int j=0; j<i; j++){
		for(int k=0; k<i; k++)
			matrix[j][k] = 0;
	}
	
	return matrix;
}

}

