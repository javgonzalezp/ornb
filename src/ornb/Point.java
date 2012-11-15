package ornb;

public class Point {
	float bin;
	int frequency;
	
	public Point(double d, int frequency){
		this.bin = (float) d;
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
