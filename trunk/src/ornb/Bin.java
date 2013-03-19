package ornb;

/**
 * Class that saves the data of a bin and its frequency in a histogram
 * @author javgonzalez
 *
 */
public class Bin {
	/**
	 * The value of the bin
	 */
	double bin;
	/**
	 * The frequency of values that the bin contains
	 */
	int frequency;
	
	public Bin(double d, int frequency){
		this.bin = d;
		this.frequency = frequency;
	}
	/**
	 * Method that sets the value of the bin
	 * @param newBin
	 */
	public void setBin(double newBin){
		bin = newBin;
	}
	/**
	 * Method that returns the value of the bin
	 * @return bin
	 */
	public double getBin(){
		return bin;
	}
	/**
	 * Method that adds one element to the total frequency of the bin
	 */
	public void addFrequency(){
		frequency++;
	}
	
	public void minusFrequency(){
		frequency--;
	}
	
	/**
	 * Method that returns the frequency of the bin
	 * @return frequency
	 */
	public int getFrequency(){
		return frequency;
	}
}
