package ornb;

import java.util.ArrayList;

import weka.core.Utils;

public class Forest {
	ArrayList<NaiveBayes> forest;
	int numNB;
	String[] classes = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
	String[] attributes = {"sepallength", "sepalwidth", "petallength", "petalwidth"};
	
	public Forest(int numNB){
		forest = new ArrayList<NaiveBayes>();
		this.numNB = numNB;
	}
	
	public void initializeForest(){
		for(int i=0; i<numNB; i++)
			forest.add(new NaiveBayes(classes, attributes));
	}
	
	public void addElement(String element){
		for(int i=0; i<numNB; i++){
			int a = (int) (Math.random() * numNB);
			System.out.println("NB: "+a);
			NaiveBayes nb = forest.get(a);
			nb.addElement(element);
		}

	}

	public double[] classify(String element) throws Exception {
		double [] sums = new double [classes.length], newProbs;
		
		for (int i = 0; i < numNB; i++) {
    		//se obtienen las probabilidades del forest de NB
    		newProbs = forest.get(i).distributionForInstance(element);
    		for (int j = 0; j < newProbs.length; j++)
    			//se suman al contador correspondiente 
    			sums[j] += newProbs[j];
	    }
		
	    if (Utils.eq(Utils.sum(sums), 0)) {
	    	return sums;
	    } else {
	    	Utils.normalize(sums);
	    	return sums;
	    }
	}
	/*
	public double[] distributionForInstance(Instance instance) throws Exception {

		double [] sums = new double [classes.length], newProbs; 
	    
	    for (int i = 0; i < numNB; i++) {
    		//se obtienen las probabilidades del forest de NB
    		newProbs = forest.get(i).distributionForInstance(instance);
    		for (int j = 0; j < newProbs.length; j++)
    			//se suman al contador correspondiente 
    			sums[j] += newProbs[j];
	    }
	    if (instance.classAttribute().isNumeric() == true) {
	    	sums[0] /= (double)numNB;
	    	return sums;
	    } else if (Utils.eq(Utils.sum(sums), 0)) {
	    	return sums;
	    	} else {
	    		Utils.normalize(sums);
	    		return sums;
	    	}
	    //obtengo el maximo del sums para definir cual es la clase que se clasifico
	}*/
	
}
