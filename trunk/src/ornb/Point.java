package ornb;

public class Point {
	float bin;
	int frequency;
	
	public Point(float bin, int frequency){
		this.bin = bin;
		this.frequency = frequency;
	}
	
	public void setBin(float newBin){
		bin = newBin;
	}
	
	public float getBin(){
		return bin;
	}
	
	public void addFrequency(){
		frequency++;
	}
	
	public int getFrequency(){
		return frequency;
	}
}
