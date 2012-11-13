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
	
	public void updateHistogram(float f){
		Point p = new Point(f, 1);
		int pos = searchBinForPoint(p);
		
		if(pos != -1)
			bins.get(pos).addFrequency();
		else{
			bins.add(p);
			sortHistogram();
			replaceBins(findMinimumDifference());
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
		bins.set(pos-1, p);
	}
	
}
