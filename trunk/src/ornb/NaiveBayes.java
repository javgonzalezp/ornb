package ornb;

import java.util.ArrayList;
import weka.core.Utils;

public class NaiveBayes {
	int numClasses, numAttributes;
	ArrayList<Histogram> histograms;
	String[] classes;
	String[] attributes;
	
	public NaiveBayes(String[] classes, String[] attributes){
		this.numClasses = classes.length;
		this.numAttributes = attributes.length;
		histograms = new ArrayList<Histogram>();
		this.classes = classes;
		this.attributes = attributes;
		initializeHistograms();
	}
	
	public void initializeHistograms(){
		for(int i=0; i<numClasses*numAttributes; i++)
			histograms.add(new Histogram(5));
	}
	
	public void addElement(String element){
		//dividir el elemento dependiendo de la clase y atributo y el valor
		String[] s = element.split(",");
		String _class = s[4];
		
		for(int i=0; i<classes.length; i++){
			if(classes[i].equalsIgnoreCase(_class)){
				for(int j=0; j<attributes.length; j++){
					histograms.get((numAttributes*i)+j).updateHistogram(Double.parseDouble(s[j]));
				}
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
	
	  public double [] distributionForInstance(String element) 
	    throws Exception {
		  String[] s = element.split(",");

	    double [] probs = getProbability();

	    for(int i=0; i<attributes.length; i++){
	    	double att = Double.parseDouble(s[i]);
	    	double temp = 0, max = 0;
	    	for (int j = 0; j < classes.length; j++) {
	    		Histogram h = histograms.get(i+numAttributes*j);
	    		//las multiplicaciones de los mean y stdv
	    		//distribucion normal
	    		double aux = -(Math.pow(att-h.getMean(),2)/2*Math.pow(h.getStandardDeviation(), 2));
	    		double aux2= 1/Math.sqrt(2*Math.PI*Math.pow(h.getStandardDeviation(), 2));
	    		temp = aux2*Math.exp(aux);
	    		//temp = Math.max(1e-75, Math.pow(m_Distributions[attIndex][j].
	              //                            getProbability(instance.value(attribute)), 
	                //                          m_Instances.attribute(attIndex).weight()));
	    		//se multiplica con la prob del elemento
	    		probs[j] *= temp;
	    		if (probs[j] > max) 
	    			max = probs[j];
	    	}
	    	if ((max > 0) && (max < 1e-75)) { // Danger of probability underflow
	    		for (int j = 0; j < classes.length; j++) {
	    			probs[j] *= 1e75;
	    		}
	    	}
	    }

	    // Display probabilities*/
	    Utils.normalize(probs);
	    return probs;
	  }
}
