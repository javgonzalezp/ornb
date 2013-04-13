package ornb_ol;
import java.io.IOException;
import java.util.ArrayList;
import onb.Window;

import org.apache.commons.math3.distribution.NormalDistribution;

import weka.core.Instance;
import weka.core.Utils;

/**
 * Class that manages the ONB algorithm and all its functions
 * @author javgonzalez
 *
 */

public class ONB {
	int numClasses, numAttributes, c = 0;
	public double tamVentana, minEntrenamiento, contEntrenamiento=0, limite, aciertos=0, errores=0, total=0, total_instancias=0;
	ArrayList<Histogram> histograms;
	String[] classes, attributes;
	Window referencia, actual;
	double[] sum;
	boolean ready = false;
	
	public ONB(String[] classes, String[] attributes, double tamVentana, double minEntrenamiento, double limite) throws IOException{
		this.numClasses = classes.length;
		this.numAttributes = attributes.length;
		histograms = new ArrayList<Histogram>();
		this.classes = classes;
		this.attributes = attributes;
		this.tamVentana = tamVentana;
		referencia = new Window(tamVentana, true, classes);
		actual = new Window(tamVentana, false, classes);
		this.minEntrenamiento = minEntrenamiento;
		this.limite = limite;
		this.sum = new double[numClasses];
		initializeHistograms();
	}
	
	/**
	 * Method that initializes the confusion matrix for the evaluation
	 * @param numClasses
	 * @return the confusion matrix with the values equal to zero
	 */
	public static double[][] initializeConfusionMatrix(int i) {
		double[][] matrix = new double[i][i];
		
		for(int j=0; j<i; j++){
			for(int k=0; k<i; k++)
				matrix[j][k] = 0;
		}
		
		return matrix;
	}
	
	/**
	 * Method that initializes the array with all the histograms for the dataset
	 */
	public void initializeHistograms(){
		for(int i=0; i<numClasses*numAttributes; i++)
			histograms.add(new Histogram(-1));
	}

	/**
	 * Method that reads an instance of the dataset and decides what is the next step, if it uses it to train
	 * the classifier or does the classification of the instance
	 * @param instance
	 * @throws Exception
	 */
	public double[] readInstance(Instance instance) throws Exception{
		double[] a = null;
		total++;
		if(contEntrenamiento<minEntrenamiento){
			addElement(instance.toString(), instance.stringValue(instance.numValues()-1));
			for(int i=0; i<classes.length; i++){
				if(classes[i].equalsIgnoreCase(instance.stringValue(instance.numValues()-1)))
					sum[i]++;
			}
			contEntrenamiento++;
		}
		else{
			ready = true;
			total_instancias++;
			a = distributionForInstance(instance.toString(), numAttributes);
			if(instance.classValue()==(double)Utils.maxIndex(a)){
				referencia.addHit();
				actual.addHit();
				aciertos++;
				addElement(instance.toString(), instance.stringValue(instance.numValues()-1));
				for(int i=0; i<classes.length; i++){
					if(classes[i].equalsIgnoreCase(instance.stringValue(instance.numValues()-1)))
						sum[i]++;
				}
				contEntrenamiento++;
			}
			else{
				referencia.addMiss();
				actual.addMiss();
				errores++;
			}
			referencia.updateProbs();
			actual.updateProbs();
			double entropia_referencia = referencia.getEntropia();
			double entropia_actual = actual.getEntropia();
			double diff = Math.abs(Math.abs(entropia_actual)-Math.abs(entropia_referencia));
			if(diff>limite){
				ready=false;
				c++;
				contEntrenamiento = 0;
				referencia.resetCounters();
				actual.resetCounters();
				for(int i=0; i<sum.length; i++){
					sum[i]=0;
				}
				histograms = new ArrayList<Histogram>();
				initializeHistograms();
			}
		}
		return a;
	}
	
	/**
	 * Method that adds an instance to the corresponding histogram
	 * @param element
	 * @param _class
	 */
	public void addElement(String element, String c){
		String[] s = element.split(",");
		
		for(int i=0; i<classes.length; i++){
			if(classes[i].equalsIgnoreCase(c)){
				for(int j=0; j<s.length-1; j++)
					histograms.get((numAttributes*i)+j).updateHistogram(Double.parseDouble(s[j]));
				break;
			}
		}
	}
	
	/**
	 * Method that calculates the mean of a particular histogram
	 * @param _attribute
	 * @param _class
	 * @return the value of the mean for a histogram
	 */
	public double calculateMean(String _attribute, String _class){
		double m = 0;
		for(int i=0; i<classes.length; i++){
			if(classes[i].equalsIgnoreCase(_class)){
				for(int j=0; j<attributes.length; j++){
					if(attributes[j].equalsIgnoreCase(_attribute)){
						m = histograms.get((numAttributes*i)+j).getMean();
						i=classes.length;
						break;
					}
				}
			}
		}
		return m;
	}
	
	/**
	 * Method that calculates the standard deviation for a particular histogram
	 * @param _attribute
	 * @param _class
	 * @return the value of the standard deviation for a histogram
	 */
	public double calculateStandarDeviation(String _attribute, String _class){
		double sd = 0;
		for(int i=0; i<classes.length; i++){
			if(classes[i].equalsIgnoreCase(_class)){
				for(int j=0; j<attributes.length; j++){
					if(attributes[j].equalsIgnoreCase(_attribute)){
						sd = histograms.get((numAttributes*i)+j).getStandardDeviation();
						i=classes.length;
						break;
					}
				}
			}
		}
		return sd;
	}
	
	/**
	 * Method that calculates the distribution of the model trained by the classifier for a instance
	 * @param element
	 * @param features
	 * @return an array with the posterior probabilities of the instance
	 * @throws Exception
	 */
	public double [] distributionForInstance(String element, int features) throws Exception {
	  String[] s = element.split(",");

	  double [] probs = getProbs();
	  for(int i=0; i<attributes.length; i++){
			double att = Double.parseDouble(s[i]);
			double temp = 0;
			for (int j = 0; j < classes.length; j++) {
				Histogram h = histograms.get(i+numAttributes*j);
				double stddev = h.getStandardDeviation();
				NormalDistribution n;
				if(stddev!=0.0){
					n = new NormalDistribution(h.getMean(), stddev);
					temp = n.density(att);
				}
	    		else{
	    			h.addBins(h.getMean());
	    			stddev = h.getStandardDeviation();
	    			n = new NormalDistribution(h.getMean(), stddev);
	    			temp = n.density(att);
	    			if(temp==0.0)
	    				temp=0.00000001;
	    		}
				probs[j] *= temp;
			    if (Double.isNaN(probs[j])) {
				    throw new Exception("NaN returned from estimator for attribute " + element);
					}
		    	}
		    }
		
		    return probs;
    	  }

	/**
	 * Method that returns an array with the prior values of the trained instances for the classifier
	 * @return an array with the prior probabilities 
	 */
	  public double[] getProbs() {
		  double[] retProbs = new double[numClasses];
		  for(int i=0; i<retProbs.length; i++)
			  retProbs[i] = sum[i]/contEntrenamiento;
		  
		  return retProbs;
	}

}
