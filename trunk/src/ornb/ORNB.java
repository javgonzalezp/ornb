package ornb;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import weka.core.Utils;

public class ORNB{
  
  /**
   * Main method for this class.
   *
   * @param argv the options
 * @throws Exception 
   */
  public static void main(String[] argv) throws Exception{
	  if (argv.length != 7) {
		  System.out.println ("Uso correcto: java -jar archivoDeTraining archivoDeTesting numClases numAtributos numBins numNB features");
		  System.exit(0);
		 }
	  String trFile = argv[0];
	  String testFile = argv[1];
	  int numClasses = Integer.parseInt(argv[2]);
	  int numAttributes = Integer.parseInt(argv[3]);
	  int numBins = Integer.parseInt(argv[4]);
	  int numNB = Integer.parseInt(argv[5]);
	  double features = Math.ceil(Double.parseDouble(argv[6])/10*numAttributes);
/*		DataSource loader;
		Instances data;
	
		loader = new DataSource("glass.arff");
		data = loader.getDataSet();
		
		if (data.classIndex() == -1)
			   data.setClassIndex(data.numAttributes()-1);

		//obtain the attributes of the arff
		String[] attributes = new String [data.numAttributes()-1];
		for(int i=0; i<data.numAttributes()-1; i++)
			attributes[i]=data.attribute(i).name();

		//obtain the classes of the arff
		String[] classes = new String [data.numClasses()];
		for(int i=0; i<data.numClasses(); i++)
			classes[i]=data.attribute(data.numAttributes()-1).value(i);

*/
		//obtain the attributes of the arff
		String[] attributes = new String [numAttributes];
		for(int i=1; i<=numAttributes; i++)
			attributes[i-1]=Integer.toString(i);

		//obtain the classes of the arff
		String[] classes = new String [numClasses];
		for(int i=1; i<=numClasses; i++)
			classes[i-1]=Integer.toString(i);

		Forest f = new Forest(numNB, classes, attributes, numBins);
   		//Predictions(nb, data);

		f.initializeForest();

		//letter.scale.tr (tr)
		File file = new File(trFile);
		BufferedReader bufRdr = new BufferedReader(new FileReader(file));
		String line = null;
		while ((line = bufRdr.readLine()) != null){
			String[] s = line.split(" ");
			f.addElement(line, s[0]);
		}
		
		//reviasr si no hay histogramas vacios
		
		int[][] matrix = initializeConfusionMatrix(numClasses);
	    //testing con iris
		//letter.scale.t (testing)
		double acc = 0.0;
		file = new File(testFile);
		bufRdr = new BufferedReader(new FileReader(file));
		line = null;
		int count=0;
		while ((line = bufRdr.readLine()) != null){
			String[] s = line.split(" ");
			double[] a = f.classify(line, (int) features);
			if(Integer.parseInt(s[0])-1==(double)Utils.maxIndex(a))
	    		 acc++;
			matrix[Utils.maxIndex(a)][Integer.parseInt(s[0])-1]++;
			count++;
		}
		System.out.println("With the values #Bins: "+numBins+" #NB: "+numNB+" Features: "+(int) features);
		System.out.println("Accurracy: "+acc/count);
		System.out.println("----------------------------------------------");
	    //printConfusionMatrix(matrix, numClasses);

		//entrenamiento con iris
/*	    Enumeration enu = data.enumerateInstances();
	    //for(int j=0; j<100; j++){
	    	//enu = data.enumerateInstances();
		    while (enu.hasMoreElements()) {
		    	 Instance instance = (Instance) enu.nextElement();
		    	 f.addElement(instance.toString(), instance.stringValue(instance.numValues()-1));
		    }
	    //}
		
		int[][] matrix = initializeConfusionMatrix(numClasses);
	    //testing con iris
		//letter.scale.t (testing)
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
	    //System.out.println("Accurracy: "+acc/data.numInstances());
	    printConfusionMatrix(matrix, numClasses);
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
	      System.out.println("Mean: "+h.getMean()+" Standard Deviation: "+h.getStandarDeviation());
	      System.out.println("Puntos menores que 15: " + h.sumProcedure(15));
	      */
  }

@SuppressWarnings("unused")
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

