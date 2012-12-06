package ornb;

import java.util.ArrayList;

import weka.core.Utils;

public class Forest {
	ArrayList<NaiveBayes> forest;
	int numNB;
	String[] classes;
	String[] attributes;
	
	public Forest(int numNB, String[] classes, String[] attributes){
		forest = new ArrayList<NaiveBayes>();
		this.classes = classes;
		this.attributes = attributes;
		this.numNB = numNB;
	}
	
	public void initializeForest(){
		for(int i=0; i<numNB; i++)
			forest.add(new NaiveBayes(classes, attributes));
	}
	
	public void addElement(String element, String _class){
		for(int i=0; i<numNB; i++){
			int a = (int) (Math.random() * 2);
			NaiveBayes nb = forest.get(i);
			if(a==1)
				nb.addElement(element, _class);
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
//		    Utils.normalize(sums);
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
