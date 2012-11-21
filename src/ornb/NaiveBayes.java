package ornb;

import java.util.ArrayList;

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
	
	public void addElement(String c, String a, String e){
		//dividir el elemento dependiendo de la clase y atributo y el valor
		
		String _class = c, _attribute = a;
		double value = Double.parseDouble(e);
		
		for(int i=0; i<classes.length; i++){
			if(classes[i].equalsIgnoreCase(_class)){
				for(int j=0; j<attributes.length; j++){
					if(attributes[j].equalsIgnoreCase(_attribute)){
						histograms.get((numAttributes*i)+j).updateHistogram(value);
						break;
					}
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
						j=attributes.length;
						i=classes.length;
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
						sd = histograms.get((numAttributes*i)+j).getStandarDeviation();
						j=attributes.length;
						i=classes.length;
					}
				}
			}
		}
		return sd;
	}
	
}
