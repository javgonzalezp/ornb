package onb;

import java.util.Enumeration;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

	/**
	 * @param args
	 */
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
