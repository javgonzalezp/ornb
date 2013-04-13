package onb;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;

import org.apache.commons.math3.distribution.NormalDistribution;

import ornb_ol.Evaluation;
import ornb_ol.Histogram;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Class that manages the ONB algorithm and all its functions
 * @author javgonzalez
 *
 */

public class ONB {
	int numClasses, numAttributes, changes = 0;
	double tamVentana, minEntrenamiento, contEntrenamiento=0, limite, aciertos=0, errores=0, total=0, total_instancias=0, m_ClassPriorsSum;
	ArrayList<Histogram> histograms;
	String[] classes, attributes;
	Window referencia, actual;
	double[] sum, m_ClassPriors;
	FileWriter writerEI, writerVI;
	double[][] matrix;
	Evaluation ev;
	
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
		writerEI = new FileWriter("entropia-instancias.dat");
		writerEI.append("Entropía actual    Instancia\n");
		writerVI = new FileWriter("ventana-instancias.dat");
		writerVI.append("Ventana actual    Instancia\n");
		
		m_ClassPriors = new double[numClasses];
		for (int i = 0; i < numClasses; i++) {
			m_ClassPriors[i] = 1;
		}
		m_ClassPriorsSum = numClasses;
		matrix = initializeConfusionMatrix(numClasses);
		ev = new ornb_ol.Evaluation(0, numClasses, m_ClassPriors, m_ClassPriorsSum, classes);
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
	public void readInstance(Instance instance) throws Exception{
		total++;
		if(contEntrenamiento<minEntrenamiento){
			/**
			 * Adds the element to the histograms and then adds the class value to the prior sums of the dataset
			 * Finally adds the class of the instance to de variables used for the final evaluation
			 */
			addElement(instance.toString(), instance.stringValue(instance.numValues()-1));
			for(int i=0; i<classes.length; i++){
				if(classes[i].equalsIgnoreCase(instance.stringValue(instance.numValues()-1)))
					sum[i]++;
			}
			contEntrenamiento++;
			//saveToFiles(0.0, 0.0, total);

			m_ClassPriors[(int) instance.classValue()] += instance.weight();
	        m_ClassPriorsSum += instance.weight();
		}
		else{
			/**
			 * Classifies the instance, updates the values of the windows and analyzes if Concept Drift happened or
			 * not, and then reset the values  
			 */
			total_instancias++;
			double[] a = distributionForInstance(instance.toString(), numAttributes);
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
				
				m_ClassPriors[(int) instance.classValue()] += instance.weight();
		        m_ClassPriorsSum += instance.weight();
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
			saveToFiles(entropia_actual, actual.total, total);
			if(diff>limite){
				changes++;
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
			ev.updateValues(total_instancias, m_ClassPriors, m_ClassPriorsSum);
			ev.updateNumericScores(a, instance.classValue(), instance.weight());
			ev.setPredictions(instance, a);
	    	matrix[Utils.maxIndex(a)][(int) instance.classValue()]++;
		}
		
	}
	
	/**
	 * Method that saves the values of the entropy and the window size for each instance read of the classifier
	 * @param diff
	 * @param tamaño
	 * @param total2
	 * @throws IOException
	 */
	private void saveToFiles(double diff, double tamaño, double total2) throws IOException {
		writerEI.append(diff+"    "+total2+"\n");
		writerVI.append(tamaño+"    "+total2+"\n");
	}

	/**
	 * Method that adds an instance to the corresponding histogram
	 * @param element
	 * @param _class
	 */
	public void addElement(String element, String _class){
		String[] s = element.split(",");
		
		for(int i=0; i<classes.length; i++){
			if(classes[i].equalsIgnoreCase(_class)){
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
	public double calculateStandardDeviation(String _attribute, String _class){
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
			  if (Double.isNaN(probs[j]))
				  throw new Exception("NaN returned from estimator for attribute " + element);

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

	public static void main(String[] argv) throws Exception{
		  if (argv.length != 4) {
			  System.out.println ("Uso correcto: java -jar archivo minEntrenamiento tamVentana difEntropia");
			  System.exit(0);
			 }
		  String file = argv[0];
		  int minEntrenamiento = Integer.parseInt(argv[1]);
		  int tamVentana = Integer.parseInt(argv[2]);
		  double difEntropia = Double.parseDouble(argv[3]);
   
		  DataSource loader;
	  	  Instances data;
	  	
		  loader = new DataSource(file);
		  data = loader.getDataSet();
			
		  if (data.classIndex() == -1)
			  data.setClassIndex(data.numAttributes()-1);

		  String[] attributes = new String [data.numAttributes()-1];
		  for(int i=0; i<data.numAttributes()-1; i++)
			  attributes[i]=data.attribute(i).name();

			//obtain the classes of the arff
		  String[] classes = new String [data.numClasses()];
		  for(int i=0; i<data.numClasses(); i++)
			  classes[i]=data.attribute(data.numAttributes()-1).value(i);
			
		  ONB onb = new ONB(classes, attributes, tamVentana, minEntrenamiento, difEntropia);
		  @SuppressWarnings("rawtypes")
		  Enumeration enu = data.enumerateInstances();

		  while (enu.hasMoreElements()) {
			  Instance instance = (Instance) enu.nextElement();
			  onb.readInstance(instance);
		  }
		  onb.writerEI.flush();
		  onb.writerEI.close();
		  onb.writerVI.flush();
		  onb.writerVI.close();
		  onb.ev.setCorrect(onb.aciertos);
		  onb.ev.setMatrix(onb.matrix);
		  System.out.println(onb.ev.toString());
		  System.out.println("Total Instancias Clasificadas: "+onb.total_instancias);
		  System.out.println("Número de cambios: "+onb.changes);
	  }
}
