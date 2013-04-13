package onb;

/**
 * Class that manages all the characteristics for the window used in the detection of Concept Drift
 * @author javgonzalez
 *
 */
public class Window {
	public double total=0, tamaño, aciertos=0, errores=0, probAciertos, probErrores;
	String[] classes;
	int[] entropia;
	boolean referencia = false;
	
	public Window(double tamVentana, boolean referencia, String[] classes){
		this.tamaño = (int) tamVentana;
		this.referencia = referencia;
		this.classes = classes;
		entropia = new int[classes.length];
	}
	
	/**
	 * Method that adds a hit to the corresponding counter when the classifier correctly classifies an instance
	 */
	public void addHit(){
		if(referencia){
			if(total<tamaño){
				aciertos++;
				addCount();
			}
		}
		else{
			aciertos++;
			addCount();
		}
	}
	
	/**
	 * Method that adds an error to the corresponding counter when the classifier incorrectly classifies an instance
	 */
	public void addMiss(){
		if(referencia){
			if(total<tamaño){
				errores++;
				addCount();
			}
		}
		else{
			errores++;
			addCount();
		}
		
	}
	
	/**
	 * Method that updates the probabilities of the hits and errors counters of the window
	 */
	public void updateProbs(){
		probAciertos=aciertos/total;
		probErrores=errores/total;
	}
	
	/**
	 * Method that adds an element to the total of instances counter for the window
	 */
	public void addCount(){
		total++;
	}
	
	/**
	 * Method that resets all the values of the hit, error and total counters of the window
	 */
	public void resetCounters(){
		aciertos=0;
		errores=0;
		total=0;
	}
	
	/**
	 * Method that returns the hit probability of the window
	 * @return the probability of hits of the window
	 */
	public double getProbAciertos(){
		return probAciertos;
	}
	
	/**
	 * Method that returns the error probability of the window
	 * @return the probability of errors of the window
	 */
	public double getProbErrores(){
		return probErrores;
	}

	/**
	 * Method that calculates the entropy of the window
	 * @return the value of the entropy for the window
	 */
	public double getEntropia() {
		double aux1=-probAciertos*(Math.log(probAciertos)/Math.log(2));
		double aux2=-probErrores*(Math.log(probErrores)/Math.log(2));
		if(probAciertos==0)
			aux1=0;
		if(probErrores==0)
			aux2=0;
		return aux1+aux2;
	}
}
