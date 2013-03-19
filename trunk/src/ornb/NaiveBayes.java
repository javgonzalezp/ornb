package ornb;

import java.util.ArrayList;
import java.util.Enumeration;

import org.apache.commons.math3.distribution.NormalDistribution;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.estimators.DiscreteEstimator;
import weka.estimators.Estimator;
import weka.estimators.KernelEstimator;
import weka.estimators.NormalEstimator;

public class NaiveBayes {
	int numClasses, numAttributes, numBins;
	ArrayList<Histogram> histograms;
	String[] classes;
	String[] attributes;
	Instances m_Instances;
	Estimator [][] m_Distributions;
	Estimator m_ClassDistribution;
	static final double DEFAULT_NUM_PRECISION = 0.01;
	boolean m_UseKernelEstimator = false;
	
	public NaiveBayes(String[] classes, String[] attributes, int numBins){
		this.numClasses = classes.length;
		this.numAttributes = attributes.length;
		histograms = new ArrayList<Histogram>();
		this.classes = classes;
		this.attributes = attributes;
		this.numBins = numBins;
		initializeHistograms();
	}
	
	public void initializeHistograms(){
		for(int i=0; i<numClasses*numAttributes; i++)
			histograms.add(new Histogram(numBins));
	}
	
	public void addElement(String element, String c){
		//dividir el elemento dependiendo de la clase y atributo y el valor
		String[] s = element.split(",");
		//String[] s = element.split(" ");
		String _class = c;
		
		if(classes[0].equalsIgnoreCase(_class))	
			_class = c;
		
		for(int i=0; i<classes.length; i++){
			if(classes[i].equalsIgnoreCase(_class)){
				//for(int j=1; j<s.length; j++){
				for(int j=0; j<s.length-1; j++){
					//String[] aux = s[j].split(":");
					histograms.get((numAttributes*i)+j).updateHistogram(Double.parseDouble(s[j]));
				}
				break;
			}
		}
	}
	
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
	
	public double[] getProbability(){
		double [] probs = new double[classes.length];
		double total_sum=0;
		for (int j = 0; j < histograms.size(); j++)
			total_sum+=histograms.get(j).getTotalFrequencies();
		
		for(int i=0; i<classes.length;i++){
			double sum=0;
			for(int j=0; j<attributes.length; j++){
				sum+=histograms.get(numAttributes*i+j).getTotalFrequencies();
			}
			probs[i]=sum/total_sum;
		}
		
		return probs;
	}
	
	public double [] distributionForInstance(String element, int features) 
	    throws Exception {
		  String[] s = element.split(",");
		  //String[] s = element.split(" ");

	    double [] probs = getProbability();
//		double [] probs = new double[classes.length];
	    
//	    ArrayList<String> v = new ArrayList<String>();
	    for(int i=0; i<classes.length; i++)
	    	probs[i]=1;
	    /*
	    int[] values = new int[features];
	    for(int i=0; i<features; i++)
	    	values[i]=-1;
	    
	    int aux=0;
	    while(aux<features){
	    	int z = (int) (Math.random() * v.size());
    		values[aux]=Integer.parseInt(v.get(z));
    		aux++;
    		v.remove(z);
	    }
	    */
	    for(int i=0; i<attributes.length; i++){
	    	//String[] a = s[values[i]].split(":");
	    	double att = Double.parseDouble(s[i]);
	    	double temp = 0;
	    	for (int j = 0; j < classes.length; j++) {
	    		Histogram h = histograms.get(i+numAttributes*j);
	    		double stddev = h.getStandardDeviation();
	    		//las multiplicaciones de los mean y stdv
	    		//
	    		NormalDistribution n;
	    		if(stddev!=0.0){
//	    			NormalDistribution n = new NormalDistribution(h.getMean(), stddev);
	    			n = new NormalDistribution(h.getMean(), stddev);
	    			temp = n.density(att);
//	    			double aux1 = n.density(1);
//	    			double aux2 = n.density(2);
//	    			double aux3 = n.density(3);
//	    			double aux5 = h.getMean();
//	    			double aux6 = aux1+aux2;
//	    			double aux7 = aux1+aux2+aux3;
//	    			double aux4 = 1;
	    		}
	    		else{
	    			h.addBins(h.getMean());
	    			stddev = h.getStandardDeviation();
	    			n = new NormalDistribution(h.getMean(), stddev);
	    			temp = n.density(att);
	    			if(temp==0.0)
	    				temp=0.00000001;
////	    			if(att==h.getMean()){
////	    				temp=1.6;
//	    				h.addBin(h.getMean());
//	    				stddev = h.getStandardDeviation();
//	    				n = new NormalDistribution(h.getMean(), stddev);
//		    			temp = n.density(att);
////	    			}
//	    				
////	    			else
////	    				temp=0.001;
	    		}
	    		//distribucion normal
	    		//double aux = -(Math.pow(att-h.getMean(),2)/2*Math.pow(h.getStandardDeviation(), 2));
	    		//double aux2= 1/Math.sqrt(2*Math.PI*Math.pow(h.getStandardDeviation(), 2));
	    		//temp = aux2*Math.exp(aux);
	    		//temp = Math.max(1e-75, Math.pow(m_Distributions[attIndex][j].
	              //                            getProbability(instance.value(attribute)), 
	                //                          m_Instances.attribute(attIndex).weight()));
	    		//if(Double.isInfinite(temp))
	    			//temp=1.0;
	    		//se multiplica con la prob del elemento
	    		
	    		probs[j] *= temp;
//	    		if (probs[j] > max) 
//	    			max = probs[j];
    		    if (Double.isNaN(probs[j])) {
    			    throw new Exception("NaN returned from estimator for attribute " + element);
    			}
	    	}
	    	/*if ((max > 0) && (max < 1e-75)) { // Danger of probability underflow
	    		for (int j = 0; j < classes.length; j++) {
	    			probs[j] *= 1e75;
	    		}
	    	}*/
	    }

	    // Display probabilities*/
//	    Utils.normalize(probs);
	    return probs;
	  }

	public boolean checkValue(int[] values, int z) {
		if(z==0)
			return false;
		
		for(int i=0; i<values.length; i++){
			if(z==values[i])
				return false;
		}
		
		return true;
	}
	
}
