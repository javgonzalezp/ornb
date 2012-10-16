package ornb;
/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    NaiveBayes.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

import java.io.File;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SparseInstance;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.estimators.DiscreteEstimator;
import weka.estimators.Estimator;
import weka.estimators.KernelEstimator;
import weka.estimators.NormalEstimator;
import weka.filters.supervised.instance.Resample;

public class NB {
  static ArrayList<double[]> lista = new ArrayList<double[]>();

  public static void Predictions(Classifier nb, Instances instances) throws Exception{
	DataSource source = new DataSource(instances);
	Instances 	test;
	Instance 	inst;
	
	source.reset();
	test = source.getStructure();
	while (source.hasMoreElements(test)) {
	  inst = source.nextElement(test);
	  double[] d = nb.distributionForInstance(inst);
	  lista.add(d);
	}
  }

  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String [] argv) {
	  NaiveBayes nb = new NaiveBayes();
	  ArrayList mult= new ArrayList(); 
	  int quantity = 100;
	  int first=0,second=0;
    //runClassifier(nb, argv);
    try {
    	DataSource loader;
    	Instances data;
    	SparseInstance si=new SparseInstance(3);
    	
		loader = new DataSource("tic-tac-toe.arff");
    	//loader = new DataSource("datos.csv");
		data = loader.getDataSet();
		
   		if (data.classIndex() == -1)
   			   data.setClassIndex(data.numAttributes()-1);
   		si.setDataset(data);
		
   		/**Desde aquí tengo que hacer el ciclo para que haga los B Naive Bayes, obteniendo
   		 * las sumas para las clases correspondientes
   		 */
   		
   		for(int i=0; i<quantity; i++){
	   	    Resample rs = new Resample();
	   	    Random r = new Random();
	   	    double a = r.nextDouble();
	   	    
	   	    rs.setSampleSizePercent(a*100);
	   	    rs.setInputFormat(data);
	   	    rs.setRandomSeed(10);
	   	    
	   	    data = Resample.useFilter(data, rs);
	   		
	   		nb.buildClassifier(data);
	   		Predictions(nb, data);
	   		double[] aux = MultiplyPredictions(2);
	    	//Aquí tengo que hacer el método o encontrar la manera para retornar las predicciones
	    	//de cada uno de los elementos para ese naive bayes
	   		
	   		first+=aux[0];
	   		second+=aux[1];
	   		//System.out.println(Utils.maxIndex(aux));

	   		//System.out.println(Evaluation.evaluateModel(nb, argv));
   		}
   		System.out.println("first: "+first+" second:"+second);
      }
      catch (Exception e) {
        if (    ((e.getMessage() != null) && (e.getMessage().indexOf("General options") == -1))
            || (e.getMessage() == null) )
          e.printStackTrace();
        else
          System.err.println(e.getMessage());
      }
    
  }

	public static double[] MultiplyPredictions(int class_index) {
		double[] array = new double[class_index];
		//ArrayList<Double> array = new ArrayList<Double>();
		for(int i=0; i<class_index; i++){
			array[i]=1.0;
		}
		for(int j=0; j<lista.size(); j++){
			//System.out.println(Utils.maxIndex(lista.get(j)));
			double[] aux = lista.get(j);
			for(int i=0; i<class_index; i++){
				array[i]=array[i]*aux[i];
				//Utils.normalize(array);
			}
		}
		//Utils.normalize(array);
		return array;
	}

}

