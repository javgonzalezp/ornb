package ornb_ol;

import java.util.ArrayList;

/**
 * Class that handles all the options for the histograms in the program
 * @author javgonzalez
 *
 */
public class Histogram {
	/**
	 * ArrayList that contains all the bins of the histogram
	 */
	ArrayList<Bin> bins;
	/**
	 * The total size of the histogram
	 */
	int size;
	
	public Histogram(int size){
		if(size==-1)
			size=10;
		bins = new ArrayList<Bin>();
		this.size = size;
	}
	/**
	 * Methos that return complete ArrayList of bins of the histogram
	 * @return bins
	 */
	public ArrayList<Bin> getHistogram(){
		return bins;
	}
	/**
	 * Method that updates the histogram depending if it has to add a new element
	 * to a existing bin or if it has to create a new one 
	 * @param d
	 */
	public void updateHistogram(double d){
		Bin p = new Bin(d, 1);
		//Search in the ArrayList of bins if the new Bin is already in the array or not
		int pos = searchBinForPoint(p);
		//If it is found in the array
		if(pos != -1)
			//Simply adds the frequency to the bin
			bins.get(pos).addFrequency();
		else{
			//If not we first add the new bin to the array of bins
			bins.add(p);
			//If the bins size is bigger that size stablished
			if(bins.size()!=1 && bins.size()>size){
				//First we sort the histogram
				sortHistogram();
				//Then we find the minimum difference between to bins and then we replace 
				//the two bins for the new one
				replaceBins(findMinimumDifference());
			}
		}
	}
	/**
	 * Method that searches for the position of a bin in the array of bins of the histogram
	 * @param p
	 * @return position
	 */
	public int searchBinForPoint(Bin p){
		//The position of the array, if is not found the method returns -1
		int position = -1;
		
		for(int i=0; i<bins.size(); i++){
			if(bins.get(i).getBin() == p.getBin()){
				position = i;
				i=bins.size();
			}
		}
		
		return position;
	}
	/**
	 * Method that sorts the bins of the histogram from lowest to highest using "selection sort"
	 */
	public void sortHistogram(){
		for (int k = 0; k < bins.size(); k++){
			int min = k;
			for (int i = k; i < bins.size(); i++){
	            if (bins.get(i).getBin() < bins.get(min).getBin())
	            	min=i;
	        }
			Bin p = bins.get(k);
			bins.set(k, bins.get(min));
			bins.set(min, p);
		}
	}
	/**
	 * Method that finds the minimum difference between a pair of consecutive bins
	 * @return The position of the bin with the minimum difference
	 */
	public int findMinimumDifference(){
		double min = Float.POSITIVE_INFINITY;
		int pos = -1;
		
		for(int i=1; i<bins.size(); i++){
			if(bins.get(i).getBin()-bins.get(i-1).getBin()<min){
				min=bins.get(i).getBin()-bins.get(i-1).getBin();
				pos = i;
			}
		}
		
		return pos;
	}
	/**
	 * Method that replaces to bins in the array for one bin
	 * @param pos
	 */
	public void replaceBins(int pos){
		double bin = (bins.get(pos-1).getBin()*bins.get(pos-1).getFrequency())+(bins.get(pos).getBin()*bins.get(pos).getFrequency());
		int frequency = (bins.get(pos-1).getFrequency())+(bins.get(pos).getFrequency());
		
		Bin p = new Bin(bin/frequency, frequency);
		
		bins.remove(pos);
		bins.remove(pos-1);
		bins.add(pos-1, p);
	}
	/**
	 * Method that prints the histogram
	 */
	public void printHistogram(){
		for(int i=0; i<size; i++)
			System.out.println("Bin: "+bins.get(i).getBin()+" Frecuencia: "+bins.get(i).getFrequency());
	}
	/**
	 * Method that returns the total frequencies that the histogram contains
	 * @return
	 */
	public int getTotalFrequencies(){
		int t = 0;
		
		for(int i=0; i<bins.size(); i++)
			t+=bins.get(i).getFrequency();
		
		return t;
	}
	/**
	 * Method that calculates the mean of the histogram
	 * @return The mean of the histogram
	 */
	public double getMean(){
		double m = 0;
		double tf = getTotalFrequencies();
		
		for(int i=0; i<bins.size(); i++)
			m+=bins.get(i).getBin()*(bins.get(i).getFrequency()/tf);
		
		return m;
	}
	/**
	 * Method that calculates the standard deviation of the histogram
	 * @return The standard deviation of the histogram
	 */
	public double getStandardDeviation(){
		double stdev = 0;
		double tf = getTotalFrequencies();
		double u = getMean();
		
		for(int i=0; i<bins.size(); i++){
			//double aux = (bins.get(i).getBin()-(bins.get(i).getBin()*(bins.get(i).getFrequency()/tf)));
			double aux = (bins.get(i).getBin()-(u));
			stdev+=aux*aux*(bins.get(i).getFrequency()/tf);
		}
			
		stdev = Math.sqrt(stdev);
		
		return stdev;
	}
	
	/**
	 * Method that allows to add bins to an histogram to avoid that the standard deviation equals zero
	 * @param mean
	 */
	public void addBins(double mean) {
		Bin b = new Bin(mean-0.001, 1);
		bins.add(b);
		b = new Bin(mean, 1);
		bins.get(0).addFrequency();
		b = new Bin(mean+0.001, 1);
		bins.add(b);
		
		
	}
	
}
