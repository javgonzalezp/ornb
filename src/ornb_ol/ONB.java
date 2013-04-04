package ornb_ol;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;

import onb.Ventana;

import org.apache.commons.math3.distribution.NormalDistribution;

import ornb.Evaluation;
import ornb.Histogram;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;


public class ONB {
	int numClasses, numAttributes, c = 0;
	public double tamVentana, minEntrenamiento, contEntrenamiento=0, limite, aciertos=0, errores=0, total=0, total_instancias=0;
	ArrayList<Histogram> histograms;
	String[] classes;
	String[] attributes;
	Ventana referencia, actual;
	double[] sum;
	FileWriter writerEI;
	FileWriter writerVI;
	double[] m_ClassPriors;
	double m_ClassPriorsSum;
	double[][] matrix;
	Evaluation ev;
	
	public ONB(String[] classes, String[] attributes, double tamVentana, double minEntrenamiento, double limite) throws IOException{
		this.numClasses = classes.length;
		this.numAttributes = attributes.length;
		histograms = new ArrayList<Histogram>();
		this.classes = classes;
		this.attributes = attributes;
		this.tamVentana = tamVentana;
		referencia = new Ventana(tamVentana, true, classes);
		actual = new Ventana(tamVentana, false, classes);
		this.minEntrenamiento = minEntrenamiento;
		this.limite = limite;
		this.sum = new double[numClasses];
		initializeHistograms();
		writerEI = new FileWriter("entropia-instancias.dat");
		writerEI.append("Entropía actual    Instancia\n");
		writerVI = new FileWriter("ventana-instancias.dat");
		writerVI.append("Ventana actual    Instancia\n");
		
		  //Para la evaluacion weka
		m_ClassPriors = new double[numClasses];
		for (int i = 0; i < numClasses; i++) {
			m_ClassPriors[i] = 1;
		}
		m_ClassPriorsSum = numClasses;
		matrix = initializeConfusionMatrix(numClasses);
		ev = new ornb.Evaluation(0, numClasses, m_ClassPriors, m_ClassPriorsSum, classes);
	}
	
	public static double[][] initializeConfusionMatrix(int i) {
		double[][] matrix = new double[i][i];
		
		for(int j=0; j<i; j++){
			for(int k=0; k<i; k++)
				matrix[j][k] = 0;
		}
		
		return matrix;
	}
	
	public void initializeHistograms(){
		for(int i=0; i<numClasses*numAttributes; i++)
			histograms.add(new Histogram(-1));
	}
	//aqui se debe hacer todos los calculos para saber si se produce concept drift y para saber si se debe 
	//enviar a entrenar o clasificar la instancia correspondiente 
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
//			saveToFiles(0.0, 0.0, total);
			//EVALUACION WEKA
//			m_ClassPriors[(int) instance.classValue()] += instance.weight();
//	        m_ClassPriorsSum += instance.weight();
		}
		else{
			total_instancias++;
			a = distributionForInstance(instance.toString(), numAttributes);
//			referencia.addClass(instance.stringValue(instance.numValues()-1));
//			actual.addClass(instance.stringValue(instance.numValues()-1));
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
				
				//EVALUACION WEKA
//				m_ClassPriors[(int) instance.classValue()] += instance.weight();
//		        m_ClassPriorsSum += instance.weight();
			}
			else{
				referencia.addMiss();
				actual.addMiss();
				errores++;
			}
			referencia.updateProbs();
			actual.updateProbs();
			//calculo entropia
//			if(referencia.total==500)
//				System.out.println("hola");
			double entropia_referencia = referencia.getEntropia();
			double entropia_actual = actual.getEntropia();
			double diff = Math.abs(Math.abs(entropia_actual)-Math.abs(entropia_referencia));
//			double diff2 = Math.abs(entropia_referencia)-Math.abs(entropia_actual);
			saveToFiles(entropia_actual, actual.total, total);
			if(diff>limite){
//			if(diff>limite){
				//reseteo de los numeros
				c++;
				System.out.println(total);
				contEntrenamiento = 0;
				referencia.resetCounters();
				actual.resetCounters();
				for(int i=0; i<sum.length; i++){
					sum[i]=0;
				}
				histograms = new ArrayList<Histogram>();
				initializeHistograms();
			}
			//EVALUACION WEKA
//			ev.updateValues(total_instancias, m_ClassPriors, m_ClassPriorsSum);
//			ev.updateNumericScores(a, instance.classValue(), instance.weight());
//			ev.setPredictions(instance, a);
//	    	matrix[Utils.maxIndex(a)][(int) instance.classValue()]++;
		}
		return a;
		
	}
	
	private void saveToFiles(double diff, double tamaño, double total2) throws IOException {
		writerEI.append(diff+"    "+total2+"\n");
		writerVI.append(tamaño+"    "+total2+"\n");
	}

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
	
	public double [] distributionForInstance(String element, int features) throws Exception {
	  String[] s = element.split(",");

	  double [] probs = getProbs();
//	  for(int a=0; a<numClasses; a++)
//		  probs[a]=1;
    
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

	  public double[] getProbs() {
		  double[] retProbs = new double[numClasses];
		  for(int i=0; i<retProbs.length; i++)
			  retProbs[i] = sum[i]/contEntrenamiento;
		  
		  return retProbs;
	}

}
