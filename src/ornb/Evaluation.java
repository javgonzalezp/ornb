package ornb;

public class Evaluation {
	double correct, pct_correct, pct_incorrect, total_instances;
	double[][] m_ConfusionMatrix;
	StringBuffer output = new StringBuffer();
	
	public Evaluation(double[][] matrix, double correct, double instances){
		m_ConfusionMatrix = matrix;
		this.correct = correct;
		total_instances = instances;
	}
	
	public double correct(){
		return correct;
	}
	
	public double incorrect(){
		return total_instances - correct;
	}
	
	public double pctCorrect(){
		return correct*100/total_instances;
	}
	
	public double pctIncorrect(){
		return incorrect()*100/total_instances;
	}
	
	public final double kappa() {
	
		double[] sumRows = new double[m_ConfusionMatrix.length];
		double[] sumColumns = new double[m_ConfusionMatrix.length];
		double sumOfWeights = 0;
		for (int i = 0; i < m_ConfusionMatrix.length; i++) {
			for (int j = 0; j < m_ConfusionMatrix.length; j++) {
				sumRows[i] += m_ConfusionMatrix[i][j];
				sumColumns[j] += m_ConfusionMatrix[i][j];
		        sumOfWeights += m_ConfusionMatrix[i][j];
		    }
		}
		double correct = 0, chanceAgreement = 0;
		for (int i = 0; i < m_ConfusionMatrix.length; i++) {
			chanceAgreement += (sumRows[i] * sumColumns[i]);
		    correct += m_ConfusionMatrix[i][i];
		}
		chanceAgreement /= (sumOfWeights * sumOfWeights);
		correct /= sumOfWeights;
	
		if (chanceAgreement < 1)
			return (correct - chanceAgreement) / (1 - chanceAgreement);
		else
		    return 1;
	}
	
	public String printConfusionMatrix() {
		StringBuffer string = new StringBuffer();
		string.append("Confusion Matrix \n");
		for(int j=0; j<m_ConfusionMatrix.length; j++){
			for(int k=0; k<m_ConfusionMatrix.length; k++)
				string.append(m_ConfusionMatrix[j][k]+" ");
			string.append("\n");
		}
		return string.toString();
	}
	
	public String toString(){
		output.append("Correctly Classified Instances \t\t"+correct()+"\t\t"+pctCorrect()+"\n");
		output.append("Incorrectly Classified Instances \t\t"+incorrect()+"\t\t"+pctIncorrect()+"\n");
		output.append("Kappa statistic \t\t"+kappa()+"\n");
		output.append(printConfusionMatrix());
		return output.toString();
	}
}
