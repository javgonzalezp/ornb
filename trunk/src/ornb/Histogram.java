package ornb;

import java.util.ArrayList;

public class Histogram {
	ArrayList<Point> bins;
	int size;
	
	public Histogram(int size){
		bins = new ArrayList<Point>();
		this.size = size;
	}
	
	public ArrayList<Point> getHistogram(){
		return bins;
	}
	
	public void updateHistogram(double d){
		Point p = new Point(d, 1);
		int pos = searchBinForPoint(p);
		
		if(pos != -1)
			bins.get(pos).addFrequency();
		else{
			bins.add(p);
			if(bins.size()!=1 && bins.size()>size){
				sortHistogram();
				replaceBins(findMinimumDifference());
			}
		}
	}
	
	public int searchBinForPoint(Point p){
		int position = -1;
		
		for(int i=0; i<bins.size(); i++){
			if(bins.get(i).getBin() == p.getBin()){
				position = i;
				i=bins.size();
			}
		}
		
		return position;
	}
	
	public void sortHistogram(){
		for (int k = 0; k < bins.size(); k++){
			int min = k;
			for (int i = k; i < bins.size(); i++){
	            if (bins.get(i).getBin() < bins.get(min).getBin())
	            	min=i;
	        }
			Point p = bins.get(k);
			bins.set(k, bins.get(min));
			bins.set(min, p);
		}
	}
	
	public int findMinimumDifference(){
		float min = Float.POSITIVE_INFINITY;
		int pos = -1;
		
		for(int i=1; i<bins.size(); i++){
			if(bins.get(i).getBin()-bins.get(i-1).getBin()<min){
				min=bins.get(i).getBin()-bins.get(i-1).getBin();
				pos = i;
			}
		}
		
		return pos;
	}
	
	public void replaceBins(int pos){
		float bin = (bins.get(pos-1).getBin()*bins.get(pos-1).getFrequency())+(bins.get(pos).getBin()*bins.get(pos).getFrequency());
		int frequency = (bins.get(pos-1).getFrequency())+(bins.get(pos).getFrequency());
		
		Point p = new Point(bin/frequency, frequency);
		
		bins.remove(pos);
		bins.remove(pos-1);
		bins.add(pos-1, p);
	}
	
	public void printHistogram(){
		for(int i=0; i<size; i++)
			System.out.println("Bin: "+bins.get(i).getBin()+" Frecuencia: "+bins.get(i).getFrequency());
	}
	
	public int getTotalFrequencies(){
		int t = 0;
		
		for(int i=0; i<size; i++)
			t+=bins.get(i).getFrequency();
		
		return t;
	}
	
	public double getMean(){
		double m = 0;
		double tf = getTotalFrequencies();
		
		for(int i=0; i<size; i++)
			m+=bins.get(i).getBin()*(bins.get(i).getFrequency()/tf);
		
		return m;
	}
	
	public double getStandarDeviation(){
		double stdev = 0;
		double tf = getTotalFrequencies();
		
		for(int i=0; i<size; i++){
			double aux = (bins.get(i).getBin()-(bins.get(i).getBin()*(bins.get(i).getFrequency()/tf)));
			stdev+=aux*aux*(bins.get(i).getFrequency()/tf);
		}
			
		stdev = Math.sqrt(stdev);
		
		return stdev;
	}
	
	public int sumProcedure(double num){
		Point p = new Point(num, 0);
		int pos = -1; 
		double freq_b = 0, sum = 0;
		
		for(int i=1; i<size; i++){
			if(bins.get(i-1).getBin()<=p.getBin() && p.getBin()<bins.get(i).getBin())
				pos=i;
		}
		
		Point x = bins.get(pos-1); //i
		Point y = bins.get(pos); //i+1
		
		double aux = y.getFrequency()-x.getFrequency();
		double aux2 = y.getBin()-x.getBin();
		double aux3= p.getBin()-x.getBin();
		
		//freq_b = x.getFrequency() + ((y.getFrequency()-x.getFrequency())/y.getBin()-x.getBin())*(p.getBin()-x.getBin());
		freq_b = x.getFrequency() + (aux*aux3/aux2);
		
		sum = ((x.getFrequency() + freq_b)/2)*((p.getBin()-x.getBin())/(y.getBin()-x.getBin()));
		
		for(int j=0; j<pos-1; j++)
			sum+=bins.get(j).getFrequency();
		
		sum+=x.getFrequency()/2;
		
		return (int) sum;
	}
	
}
