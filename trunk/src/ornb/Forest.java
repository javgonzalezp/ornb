package ornb;

import java.util.ArrayList;

public class Forest {
	ArrayList<NaiveBayes> forest;
	int numNB;
	String[] classes = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
	String[] attributes = {"sepallength", "sepalwidth", "petallength", "petalwidth"};
	
	public Forest(int numNB){
		forest = new ArrayList<NaiveBayes>();
		this.numNB = numNB;
	}
	
	public void initializeForest(){
		for(int i=0; i<numNB; i++)
			forest.add(new NaiveBayes(classes, attributes));
	}
	
	public void addElement(String element){
		String[] e = element.split(",");
		
		for(int i=0; i<attributes.length; i++){
			int a = (int) (Math.random() * 9);
			NaiveBayes nb = forest.get(a);
			nb.addElement(e[4],attributes[i],e[i]);
		}

	}
}
