package ornb_ol;

import java.util.Enumeration;

import ornb_ol.Forest;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class ORNB {

	/**
	 * @param args
	 */
	public static void main(String[] argv) throws Exception{
		  if (argv.length != 5) {
			  System.out.println ("Uso correcto: java -jar archivo minEntrenamiento tamVentana difEntropia numONB");
			  System.exit(0);
			 }
		  String file = argv[0];
		  int minEntrenamiento = Integer.parseInt(argv[1]);
		  int tamVentana = Integer.parseInt(argv[2]);
		  double difEntropia = Double.parseDouble(argv[3]);
		  int numNB = Integer.parseInt(argv[4]);
 
		  DataSource loader;
	  	  Instances data;
	  	
		  loader = new DataSource(file);
		  data = loader.getDataSet();
			
		  if (data.classIndex() == -1)
			  data.setClassIndex(data.numAttributes()-1);

		  String[] attributes = new String [data.numAttributes()-1];
		  for(int i=0; i<data.numAttributes()-1; i++)
			  attributes[i]=data.attribute(i).name();

		  String[] classes = new String [data.numClasses()];
		  for(int i=0; i<data.numClasses(); i++)
			  classes[i]=data.attribute(data.numAttributes()-1).value(i);
			
		  Forest f = new Forest(numNB, classes, attributes, minEntrenamiento, tamVentana, difEntropia);
		  f.initializeForest();
		  
		  @SuppressWarnings("rawtypes")
		Enumeration enu = data.enumerateInstances();
		    //for(int j=0; j<100; j++){
		    	//enu = data.enumerateInstances();
		  while (enu.hasMoreElements()) {
			  Instance instance = (Instance) enu.nextElement();
			  f.readInstance(instance);
		  }
		  System.out.println("Aciertos: "+f.aciertos);
		  System.out.println("Total: "+f.total);
		  System.out.println("Razon: "+f.aciertos/f.total);
		  System.out.println(printConfusionMatrix(f.matrix, classes.length, classes));
	}

	public static String printConfusionMatrix(double[][] m_ConfusionMatrix, int m_NumClasses, String[] m_ClassNames) {
	    StringBuffer text = new StringBuffer();
	    char[] IDChars = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
	        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
	        'z' };
	    int IDWidth;
	    boolean fractional = false;

	    double maxval = 0;
	    for (int i = 0; i < m_NumClasses; i++) {
	      for (int j = 0; j < m_NumClasses; j++) {
	        double current = m_ConfusionMatrix[i][j];
	        if (current < 0) {
	          current *= -10;
	        }
	        if (current > maxval) {
	          maxval = current;
	        }
	        double fract = current - Math.rint(current);
	        if (!fractional && ((Math.log(fract) / Math.log(10)) >= -2)) {
	          fractional = true;
	        }
	      }
	    }

	    IDWidth = 1 + Math.max(
	        (int) (Math.log(maxval) / Math.log(10) + (fractional ? 3 : 0)),
	        (int) (Math.log(m_NumClasses) / Math.log(IDChars.length)));
	    text.append("=== Confusion Matrix ===\n").append("\n");
	    for (int i = 0; i < m_NumClasses; i++) {
	      if (fractional) {
	        text.append(" ").append(num2ShortID(i, IDChars, IDWidth - 3))
	            .append("   ");
	      } else {
	        text.append(" ").append(num2ShortID(i, IDChars, IDWidth));
	      }
	    }
	    text.append("   <-- classified as\n");
	    for (int i = 0; i < m_NumClasses; i++) {
	      for (int j = 0; j < m_NumClasses; j++) {
	        text.append(" ").append(
	            Utils.doubleToString(m_ConfusionMatrix[i][j], IDWidth,
	                (fractional ? 2 : 0)));
	      }
	      text.append(" | ").append(num2ShortID(i, IDChars, IDWidth)).append(" = ")
	          .append(m_ClassNames[i]).append("\n");
	    }
	    return text.toString();
	  }
	
	  protected static String num2ShortID(int num, char[] IDChars, int IDWidth) {

		    char ID[] = new char[IDWidth];
		    int i;

		    for (i = IDWidth - 1; i >= 0; i--) {
		      ID[i] = IDChars[num % IDChars.length];
		      num = num / IDChars.length - 1;
		      if (num < 0) {
		        break;
		      }
		    }
		    for (i--; i >= 0; i--) {
		      ID[i] = ' ';
		    }

		    return new String(ID);
		  }
}
