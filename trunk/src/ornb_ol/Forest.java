package ornb_ol;

import java.io.IOException;
import java.util.ArrayList;

import ornb_ol.ONB;

import weka.core.Instance;
import weka.core.Utils;

public class Forest {
	ArrayList<ONB> forest;
	int numNB, numBins, minEntrenamiento, tamVentana;
	double diffEntropia;
	String[] classes;
	String[] attributes;
	
	public Forest(int numNB, String[] classes, String[] attributes, int minEntrenamiento, int tamVentana, double difEntropia){
		forest = new ArrayList<ONB>();
		this.classes = classes;
		this.attributes = attributes;
		this.numNB = numNB;
		this.minEntrenamiento = minEntrenamiento;
		this.tamVentana = tamVentana;
		this.diffEntropia = difEntropia;
	}

	public void initializeForest() throws IOException{
		for(int i=0; i<numNB; i++)
			forest.add(new ONB(classes, attributes, tamVentana, minEntrenamiento, diffEntropia));
	}
	
	public void training(Instance instance) throws Exception{
		for(int i=0; i<numNB; i++){
			int random = (int) (Math.random() * 2);
			ONB onb = forest.get(i);
			if(random==1)
				onb.readInstance(instance);
		}
	}

	public double[] classify(Instance instance) throws Exception {
		double [] sums = new double [classes.length], newProbs;
		
		for (int i = 0; i < numNB; i++) {
    		//se obtienen las probabilidades del forest de NB
//    		newProbs = forest.get(i).distributionForInstance(instance.toString(), attributes.length);
			newProbs = forest.get(i).readInstance(instance);
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

	public void readInstance(Instance instance) throws Exception {
		if(allONBTrained())
			classify(instance);
		//dsps de aqui debo hacer la evaluacion
		else
			training(instance);
		
	}

	private boolean allONBTrained() {
		for(int i=0; i<numNB; i++){
			if(forest.get(i).contEntrenamiento<minEntrenamiento)
				return false;
			
		}
		return true;
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
