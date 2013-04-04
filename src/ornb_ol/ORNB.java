package ornb_ol;

import java.util.Enumeration;

import ornb_ol.Forest;

import onb.ONB;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ORNB {

	/**
	 * @param args
	 */
	public static void main(String[] argv) throws Exception{
		  if (argv.length != 5) {
			  System.out.println ("Uso correcto: java -jar archivo minEntrenamiento tamVentana difEntropia numNB");
			  System.exit(0);
			 }
		  String file = argv[0];
		  int minEntrenamiento = Integer.parseInt(argv[1]);
		  int tamVentana = Integer.parseInt(argv[2]);
		  double difEntropia = Double.parseDouble(argv[3]);
		  int numNB = Integer.parseInt(argv[4]);
		  int cont = 0;
 
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
			
		  Forest f = new Forest(numNB, classes, attributes, minEntrenamiento, tamVentana, difEntropia);
		  f.initializeForest();
		  
		  Enumeration enu = data.enumerateInstances();
		    //for(int j=0; j<100; j++){
		    	//enu = data.enumerateInstances();
		  while (enu.hasMoreElements()) {
			  Instance instance = (Instance) enu.nextElement();
			  f.readInstance(instance);
		  }
//		  System.out.println(onb.aciertos*100/onb.total_instancias);
//		  System.out.println(onb.errores*100/onb.total_instancias);
//		  onb.ev.setCorrect(onb.aciertos);
//		  onb.ev.setMatrix(onb.matrix);
//		  System.out.println(onb.ev.toString());
//		  System.out.println("Total Instancias Clasificadas: "+onb.total_instancias);
//		  System.out.println("Número de cambios: "+onb.c);
		  //}
	  }

}
