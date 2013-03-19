package onb;
import java.util.ArrayList;
import java.util.Enumeration;

import org.apache.commons.math3.distribution.NormalDistribution;

import ornb.Histogram;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;


public class ONB {
	int numClasses, numAttributes, c = 0;
	double tamVentana, minEntrenamiento, contEntrenamiento=0, limite, aciertos=0, errores=0, total=0;
	ArrayList<Histogram> histograms;
	String[] classes;
	String[] attributes;
	Ventana referencia, actual;
	double[] sum;
	
	public ONB(String[] classes, String[] attributes, double tamVentana, double minEntrenamiento, double limite){
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
	}
	
	public void initializeHistograms(){
		for(int i=0; i<numClasses*numAttributes; i++)
			histograms.add(new Histogram(-1));
	}
	//aqui se debe hacer todos los calculos para saber si se produce concept drift y para saber si se debe 
	//enviar a entrenar o clasificar la instancia correspondiente 
	public void readInstance(Instance instance) throws Exception{
		if(contEntrenamiento<minEntrenamiento){
			addElement(instance.toString(), instance.stringValue(instance.numValues()-1));
			for(int i=0; i<classes.length; i++){
				if(classes[i].equalsIgnoreCase(instance.stringValue(instance.numValues()-1)))
					sum[i]++;
			}
			contEntrenamiento++;
		}
		else{
			double[] a = distributionForInstance(instance.toString(), numAttributes);
			total++;
//			referencia.addClass(instance.stringValue(instance.numValues()-1));
//			actual.addClass(instance.stringValue(instance.numValues()-1));
			if(instance.classValue()==(double)Utils.maxIndex(a)){

				referencia.addHit();
				actual.addHit();
				aciertos++;
				addElement(instance.toString(), instance.stringValue(instance.numValues()-1));
				contEntrenamiento++;
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
//			double diff = Math.abs(entropia_actual)-Math.abs(entropia_referencia);
//			double diff2 = Math.abs(entropia_referencia)-Math.abs(entropia_actual);
			if(Math.abs(Math.abs(entropia_actual)-Math.abs(entropia_referencia))>limite){
				//reseteo de los numeros
				c++;
				contEntrenamiento = 0;
				referencia.resetCounters();
				actual.resetCounters();
			}
			
		}
		
	}
	
	public void addElement(String element, String c){
		String[] s = element.split(",");
		String _class = c;
		
		if(classes[0].equalsIgnoreCase(_class))	
			_class = c;
		
		for(int i=0; i<classes.length; i++){
			if(classes[i].equalsIgnoreCase(_class)){
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
		  Enumeration enu = data.enumerateInstances();
		    //for(int j=0; j<100; j++){
		    	//enu = data.enumerateInstances();
		  while (enu.hasMoreElements()) {
			  Instance instance = (Instance) enu.nextElement();
			  onb.readInstance(instance);
		  }
		  System.out.println(onb.aciertos*100/onb.total);
		  System.out.println(onb.errores*100/onb.total);
		  System.out.println(onb.c);
		  //}
	  }
}
