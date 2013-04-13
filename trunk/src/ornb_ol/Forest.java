package ornb_ol;

import java.io.IOException;
import java.util.ArrayList;

import ornb_ol.ONB;

import weka.core.Instance;
import weka.core.Utils;

/**
 * Class that manages all the characteristics of the forest in the ORNB classifier
 * @author javgonzalez
 *
 */
public class Forest {
	ArrayList<ONB> forest;
	int numNB, numBins, minEntrenamiento, tamVentana;
	double diffEntropia, aciertos = 0, total = 0;
	String[] classes, attributes;
	double[][] matrix;
	double [] sums;
	
	public Forest(int numNB, String[] classes, String[] attributes, int minEntrenamiento, int tamVentana, double difEntropia){
		forest = new ArrayList<ONB>();
		this.classes = classes;
		this.attributes = attributes;
		this.numNB = numNB;
		this.minEntrenamiento = minEntrenamiento;
		this.tamVentana = tamVentana;
		this.diffEntropia = difEntropia;
		sums = new double [classes.length];
		matrix = initializeConfusionMatrix(classes.length);
	}

	/**
	 * Method that initializes the ONB in the forest 
	 * @throws IOException
	 */
	public void initializeForest() throws IOException{
		for(int i=0; i<numNB; i++)
			forest.add(new ONB(classes, attributes, tamVentana, minEntrenamiento, diffEntropia));
	}
	
	/**
	 * Method that manages the training of the instances for the forest
	 * @param instance
	 * @param onb
	 * @throws Exception
	 */
	public void training(Instance instance, ONB onb) throws Exception{
		int random = (int) (Math.random() * 2);
		if(random==1 && !onb.ready)
			onb.readInstance(instance);
	}

	/**
	 * Method that manages the classification of an instance in the forest
	 * @param instance
	 * @param onb
	 * @throws Exception
	 */
	public void classify(Instance instance, ONB onb) throws Exception {
		double[] newProbs = onb.readInstance(instance);
		for (int j = 0; j < newProbs.length; j++)
			sums[j] += newProbs[j];
	}

	/**
	 * Method that decides the course of an instance in the forest, if it goes to the training of an ONB or if its
	 * classified
	 * @param instance
	 * @throws Exception
	 */
	public void readInstance(Instance instance) throws Exception {
		boolean flag = false;
		for(int i=0; i< numNB; i++){
			ONB onb = forest.get(i);
			if(onb.ready){
				classify(instance, onb);
				flag=true;
			}
			else
				training(instance, onb);
		}
		if(flag){
			if (!Utils.eq(Utils.sum(sums), 0))
				Utils.normalize(sums);

			total++;
			if(Utils.maxIndex(sums)==(int) instance.classValue())
				aciertos++;
			matrix[Utils.maxIndex(sums)][(int) instance.classValue()]++;
		}
		sums = new double [classes.length];
	}

	/**
	 * Method that initializes the confusion matrix for the evaluation
	 * @param numClasses
	 * @return the confusion matrix with the values equal to zero
	 */
	public static double[][] initializeConfusionMatrix(int numClasses) {
		double[][] matrix = new double[numClasses][numClasses];
		
		for(int i=0; i<numClasses; i++){
			for(int j=0; j<numClasses; j++)
				matrix[i][j] = 0;
		}
		
		return matrix;
	}
}
