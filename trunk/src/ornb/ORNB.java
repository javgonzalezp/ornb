package ornb;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomForest;
import weka.core.AdditionalMeasureProducer;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.SparseInstance;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.filters.supervised.instance.Resample;

/**
 <!-- globalinfo-start -->
 * Class for constructing a forest of random trees.<br/>
 * <br/>
 * For more information see: <br/>
 * <br/>
 * Leo Breiman (2001). Random Forests. Machine Learning. 45(1):5-32.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Breiman2001,
 *    author = {Leo Breiman},
 *    journal = {Machine Learning},
 *    number = {1},
 *    pages = {5-32},
 *    title = {Random Forests},
 *    volume = {45},
 *    year = {2001}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -I &lt;number of trees&gt;
 *  Number of trees to build.</pre>
 * 
 * <pre> -K &lt;number of features&gt;
 *  Number of features to consider (&lt;1=int(logM+1)).</pre>
 * 
 * <pre> -S
 *  Seed for random number generator.
 *  (default 1)</pre>
 * 
 * <pre> -depth &lt;num&gt;
 *  The maximum depth of the trees, 0 for unlimited.
 *  (default 0)</pre>
 * 
 * <pre> -print
 *  Print the individual trees in the output</pre>
 * 
 * <pre> -num-slots &lt;num&gt;
 *  Number of execution slots.
 *  (default 1 - i.e. no parallelism)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 <!-- options-end -->
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 8892 $
 */
public class ORNB 
  extends AbstractClassifier 
  implements OptionHandler, Randomizable, WeightedInstancesHandler, 
             AdditionalMeasureProducer, TechnicalInformationHandler {

  /** for serialization */
  static final long serialVersionUID = 4216839470751428698L;
  
  /** Number of trees in forest. */
  protected int m_numTrees = 10;

  /** Number of features to consider in random feature selection.
      If less than 1 will use int(logM+1) ) */
  protected int m_numFeatures = 0;

  /** The random seed. */
  protected int m_randomSeed = 1;  

  /** Final number of features that were considered in last build. */
  protected int m_KValue = 0;

  /** The bagger. */
  protected Bagging m_bagger = null;
  
  /** The maximum depth of the trees (0 = unlimited) */
  protected int m_MaxDepth = 0;
  
  /** The number of threads to have executing at any one time */
  protected int m_numExecutionSlots = 1;
  
  /** Print the individual trees in the output */
  protected boolean m_printTrees = false;
  
  static ArrayList<double[]> lista = new ArrayList<double[]>();

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {

    return  
        "Class for constructing a forest of random trees.\n\n"
      + "For more information see: \n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Leo Breiman");
    result.setValue(Field.YEAR, "2001");
    result.setValue(Field.TITLE, "Random Forests");
    result.setValue(Field.JOURNAL, "Machine Learning");
    result.setValue(Field.VOLUME, "45");
    result.setValue(Field.NUMBER, "1");
    result.setValue(Field.PAGES, "5-32");
    
    return result;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numTreesTipText() {
    return "The number of trees to be generated.";
  }

  /**
   * Get the value of numTrees.
   *
   * @return Value of numTrees.
   */
  public int getNumTrees() {
    
    return m_numTrees;
  }
  
  /**
   * Set the value of numTrees.
   *
   * @param newNumTrees Value to assign to numTrees.
   */
  public void setNumTrees(int newNumTrees) {
    
    m_numTrees = newNumTrees;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numFeaturesTipText() {
    return "The number of attributes to be used in random selection (see RandomTree).";
  }

  /**
   * Get the number of features used in random selection.
   *
   * @return Value of numFeatures.
   */
  public int getNumFeatures() {
    
    return m_numFeatures;
  }
  
  /**
   * Set the number of features to use in random selection.
   *
   * @param newNumFeatures Value to assign to numFeatures.
   */
  public void setNumFeatures(int newNumFeatures) {
    
    m_numFeatures = newNumFeatures;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String seedTipText() {
    return "The random number seed to be used.";
  }

  /**
   * Set the seed for random number generation.
   *
   * @param seed the seed 
   */
  public void setSeed(int seed) {

    m_randomSeed = seed;
  }
  
  /**
   * Gets the seed for the random number generations
   *
   * @return the seed for the random number generation
   */
  public int getSeed() {

    return m_randomSeed;
  }
  
  /**
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String maxDepthTipText() {
    return "The maximum depth of the trees, 0 for unlimited.";
  }

  /**
   * Get the maximum depth of trh tree, 0 for unlimited.
   *
   * @return 		the maximum depth.
   */
  public int getMaxDepth() {
    return m_MaxDepth;
  }
  
  /**
   * Set the maximum depth of the tree, 0 for unlimited.
   *
   * @param value 	the maximum depth.
   */
  public void setMaxDepth(int value) {
    m_MaxDepth = value;
  }
  
  /**
   * Returns the tip text for this property
   * 
   * @return            tip text for this property suitable for
   *                    displaying in the explorer/experimenter gui
   */
  public String printTreesTipText() {
    return "Print the individual trees in the output";
  }
  
  /**
   * Set whether to print the individual ensemble trees in the output
   * 
   * @param print true if the individual trees are to be printed
   */
  public void setPrintTrees(boolean print) {
    m_printTrees = print;
  }
  
  /**
   * Get whether to print the individual ensemble trees in the output
   * 
   * @return true if the individual trees are to be printed
   */
  public boolean getPrintTrees() {
    return m_printTrees;
  }

  /**
   * Gets the out of bag error that was calculated as the classifier was built.
   *
   * @return the out of bag error
   */
  public double measureOutOfBagError() {
    
    if (m_bagger != null) {
      return m_bagger.measureOutOfBagError();
    } else return Double.NaN;
  }
  
  /**
   * Set the number of execution slots (threads) to use for building the
   * members of the ensemble.
   *
   * @param numSlots the number of slots to use.
   */
  public void setNumExecutionSlots(int numSlots) {
    m_numExecutionSlots = numSlots;
  }

  /**
   * Get the number of execution slots (threads) to use for building
   * the members of the ensemble.
   *
   * @return the number of slots to use
   */
  public int getNumExecutionSlots() {
    return m_numExecutionSlots;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numExecutionSlotsTipText() {
    return "The number of execution slots (threads) to use for " +
      "constructing the ensemble.";
  }
  
  /**
   * Returns an enumeration of the additional measure names.
   *
   * @return an enumeration of the measure names
   */
  public Enumeration enumerateMeasures() {
    
    Vector newVector = new Vector(1);
    newVector.addElement("measureOutOfBagError");
    return newVector.elements();
  }
  
  /**
   * Returns the value of the named measure.
   *
   * @param additionalMeasureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  public double getMeasure(String additionalMeasureName) {
    
    if (additionalMeasureName.equalsIgnoreCase("measureOutOfBagError")) {
      return measureOutOfBagError();
    }
    else {throw new IllegalArgumentException(additionalMeasureName 
					     + " not supported (RandomForest)");
    }
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options
   */
  public Enumeration listOptions() {
    
    Vector newVector = new Vector();

    newVector.addElement(new Option(
	"\tNumber of trees to build.",
	"I", 1, "-I <number of trees>"));
    
    newVector.addElement(new Option(
	"\tNumber of features to consider (<1=int(logM+1)).",
	"K", 1, "-K <number of features>"));
    
    newVector.addElement(new Option(
	"\tSeed for random number generator.\n"
	+ "\t(default 1)",
	"S", 1, "-S"));

    newVector.addElement(new Option(
	"\tThe maximum depth of the trees, 0 for unlimited.\n"
	+ "\t(default 0)",
	"depth", 1, "-depth <num>"));
    
    newVector.addElement(new Option(
        "\tPrint the individual trees in the output", "print", 0, "-print"));
    
    newVector.addElement(new Option(
        "\tNumber of execution slots.\n"
        + "\t(default 1 - i.e. no parallelism)",
        "num-slots", 1, "-num-slots <num>"));

    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      newVector.addElement(enu.nextElement());
    }

    return newVector.elements();
  }

  /**
   * Gets the current settings of the forest.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String[] getOptions() {
    Vector        result;
    String[]      options;
    int           i;
    
    result = new Vector();
    
    result.add("-I");
    result.add("" + getNumTrees());
    
    result.add("-K");
    result.add("" + getNumFeatures());
    
    result.add("-S");
    result.add("" + getSeed());
    
    if (getMaxDepth() > 0) {
      result.add("-depth");
      result.add("" + getMaxDepth());
    }
    
    if (m_printTrees) {
      result.add("-print");
    }
    
    result.add("-num-slots");
    result.add("" + getNumExecutionSlots());
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    return (String[]) result.toArray(new String[result.size()]);
  }

  /**
   * Parses a given list of options. <p/>
   * 
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -I &lt;number of trees&gt;
   *  Number of trees to build.</pre>
   * 
   * <pre> -K &lt;number of features&gt;
   *  Number of features to consider (&lt;1=int(logM+1)).</pre>
   * 
   * <pre> -S
   *  Seed for random number generator.
   *  (default 1)</pre>
   * 
   * <pre> -depth &lt;num&gt;
   *  The maximum depth of the trees, 0 for unlimited.
   *  (default 0)</pre>
   * 
   * <pre> -print
   *  Print the individual trees in the output</pre>
   * 
   * <pre> -num-slots &lt;num&gt;
   *  Number of execution slots.
   *  (default 1 - i.e. no parallelism)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   <!-- options-end -->
   * 
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception{
    String	tmpStr;
    
    tmpStr = Utils.getOption('I', options);
    if (tmpStr.length() != 0) {
      m_numTrees = Integer.parseInt(tmpStr);
    } else {
      m_numTrees = 10;
    }
    
    tmpStr = Utils.getOption('K', options);
    if (tmpStr.length() != 0) {
      m_numFeatures = Integer.parseInt(tmpStr);
    } else {
      m_numFeatures = 0;
    }
    
    tmpStr = Utils.getOption('S', options);
    if (tmpStr.length() != 0) {
      setSeed(Integer.parseInt(tmpStr));
    } else {
      setSeed(1);
    }
    
    tmpStr = Utils.getOption("depth", options);
    if (tmpStr.length() != 0) {
      setMaxDepth(Integer.parseInt(tmpStr));
    } else {
      setMaxDepth(0);
    }
    
    setPrintTrees(Utils.getFlag("print", options));

    tmpStr = Utils.getOption("num-slots", options);
    if (tmpStr.length() > 0) {
      setNumExecutionSlots(Integer.parseInt(tmpStr));
    } else {
      setNumExecutionSlots(1);
    }
    
    super.setOptions(options);
    
    Utils.checkForRemainingOptions(options);
  }  

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    return new NaiveBayes().getCapabilities();
  }

  /**
   * Builds a classifier for a set of instances.
   *
   * @param data the instances to train the classifier with
   * @throws Exception if something goes wrong
   */
  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    m_bagger = new Bagging();
    
    //Para el caso de RF cada nodo del árbol corresponde a un RandomTree 
    //RandomTree rTree = new RandomTree();
    //Para el caso de NB cada hoja del árbol corresponde a un clasificador Naive Bayes
    NaiveBayes nb = new NaiveBayes();
    
    // set up the random tree options
    m_KValue = m_numFeatures;
    if (m_KValue < 1) m_KValue = (int) Utils.log2(data.numAttributes())+1;
    //rTree.setKValue(m_KValue);
    //rTree.setMaxDepth(getMaxDepth());

    // set up the bagger and build the forest
    m_bagger.setClassifier(nb);
    m_bagger.setSeed(100);
    m_bagger.setNumIterations(m_numTrees);
    m_bagger.setCalcOutOfBag(true);
    m_bagger.setNumExecutionSlots(m_numExecutionSlots);
    m_bagger.buildClassifier(data);
    
  }

  /**
   * Returns the class probability distribution for an instance.
   *
   * @param instance the instance to be classified
   * @return the distribution the forest generates for the instance
   * @throws Exception if computation fails
   */
  public double[] distributionForInstance(Instance instance) throws Exception {

    return m_bagger.distributionForInstance(instance);
  }

  /**
   * Outputs a description of this classifier.
   *
   * @return a string containing a description of the classifier
   */
  public String toString() {

    if (m_bagger == null) { 
      return "Random forest not built yet";
    } else {
      StringBuffer temp = new StringBuffer();
      temp.append("Random forest of " + m_numTrees
          + " trees, each constructed while considering "
          + m_KValue + " random feature" + (m_KValue==1 ? "" : "s") + ".\n"
          + "Out of bag error: "
          + Utils.doubleToString(m_bagger.measureOutOfBagError(), 4) + "\n"
          + (getMaxDepth() > 0 ? ("Max. depth of trees: " + getMaxDepth() + "\n") : (""))
          + "\n");
      if (m_printTrees) {
        temp.append(m_bagger.toString());
      }
      return temp.toString();
    }
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 8892 $");
  }
  
  public void Predictions(Classifier nb, Instances instances) throws Exception{
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
   * Main method for this class.
   *
   * @param argv the options
   */
  public static void main(String[] argv) {
	  ORNB ornb = new ORNB();
	    try {
	    	DataSource loader;
	    	Instances data;
	    	SparseInstance i=new SparseInstance(3);
	    	
			loader = new DataSource("iris.arff");
			data = loader.getDataSet();
			
	   		if (data.classIndex() == -1)
	   			   data.setClassIndex(data.numAttributes()-1);
	   		i.setDataset(data);
			
	   	    Resample rs = new Resample();
	   	    Random r = new Random();
   			double a = r.nextDouble();
	   	    
   			rs.setSampleSizePercent(a*100);
	   	    rs.setInputFormat(data);
	   	    rs.setRandomSeed(10);
	   	    
	   	    ornb.setPrintTrees(true);
	   	    ornb.buildClassifier(data);
	   		data = Resample.useFilter(data, rs);
	   		ornb.Predictions(ornb, data);
	   		System.out.println(data.toString());
	   		
			for(int j=0; j<lista.size(); j++)
				System.out.println(Utils.maxIndex(lista.get(j)));
	      }
	      catch (Exception e) {
	        if (    ((e.getMessage() != null) && (e.getMessage().indexOf("General options") == -1))
	            || (e.getMessage() == null) )
	          e.printStackTrace();
	        else
	          System.err.println(e.getMessage());
	      }
  }
}

