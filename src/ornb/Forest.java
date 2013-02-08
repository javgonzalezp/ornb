package ornb;

import java.util.ArrayList;

import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Utils;

public class Forest {
	ArrayList<NaiveBayes> forest;
	int numNB, numBins;
	String[] classes;
	String[] attributes;
	int naiveb;
	
	public Forest(int numNB, String[] classes, String[] attributes, int numBins){
		forest = new ArrayList<NaiveBayes>();
		this.classes = classes;
		this.attributes = attributes;
		this.numNB = numNB;
		this.numBins = numBins;
		naiveb = (int) (Math.random() * numNB);
	}
	
	public void initializeForest(){
		for(int i=0; i<numNB; i++)
			forest.add(new NaiveBayes(classes, attributes, numBins));
	}
	
	public void addElement(String element, String _class){
	//public void addElement(Instance instance){
//		for(int i=0; i<numNB; i++){
//			int a = (int) (Math.random() * 2);
//			NaiveBayes nb = forest.get(i);
//			if(a==1)
//				nb.addElement(element, _class);
//		}
		NaiveBayes nb = forest.get((int) (Math.random() * numNB));
//		if(naiveb>=numNB)
//			naiveb=0;
//		NaiveBayes nb = forest.get(naiveb);
		nb.addElement(element, _class);
//		naiveb++;
	}

	public double[] classify(String element, int features) throws Exception {
		double [] sums = new double [classes.length], newProbs;
		
		for (int i = 0; i < numNB; i++) {
    		//se obtienen las probabilidades del forest de NB
    		newProbs = forest.get(i).distributionForInstance(element, features);
    		for (int j = 0; j < newProbs.length; j++)
    			//se suman al contador correspondiente 
    			sums[j] += newProbs[j];
    		}
		
//	    if (Utils.eq(Utils.sum(sums), 0)) {
//	    	return sums;
//	    } else {
//	    	Utils.normalize(sums);
	    	return sums;
//	    }

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
