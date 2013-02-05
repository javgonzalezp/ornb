package ornb;

import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class Evaluation {
	double correct, pct_correct, pct_incorrect, total_instances;
	double[][] m_ConfusionMatrix;
	StringBuffer output = new StringBuffer();
	int m_NumClasses;
    protected FastVector m_Predictions;
    protected double m_SumErr, m_SumAbsErr, m_SumSqrErr, m_SumPriorAbsErr, m_SumPriorSqrErr;
    protected double[] m_ClassPriors;
    protected double m_ClassPriorsSum;
    String[] m_ClassNames;

	
	public Evaluation(double instances, int numClasses, double[] m_ClassPriors, double m_ClassPriorsSum, String[] classes){
		total_instances = instances;
		m_NumClasses = numClasses;
		this.m_ClassPriors = m_ClassPriors;
		this.m_ClassPriorsSum = m_ClassPriorsSum;
		m_ClassNames = classes;
	}
	
	public void setMatrix(double[][] matrix){
		m_ConfusionMatrix = matrix;
	}
	
	public void setCorrect(double correct){
		this.correct = correct;
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
	
	  public double precision(int classIndex) {

		    double correct = 0, total = 0;
		    for (int i = 0; i < m_ConfusionMatrix.length; i++) {
		      if (i == classIndex) {
		        correct += m_ConfusionMatrix[i][classIndex];
		      }
		      total += m_ConfusionMatrix[i][classIndex];
		    }
		    if (total == 0) {
		      return 0;
		    }
		    return correct / total;
		  }
	  
	  public double truePositiveRate(int classIndex) {

		    double correct = 0, total = 0;
		    for (int j = 0; j < m_ConfusionMatrix.length; j++) {
		      if (j == classIndex) {
		        correct += m_ConfusionMatrix[classIndex][j];
		      }
		      total += m_ConfusionMatrix[classIndex][j];
		    }
		    if (total == 0) {
		      return 0;
		    }
		    return correct / total;
		  }
	  
	  public double falsePositiveRate(int classIndex) {

		    double incorrect = 0, total = 0;
		    for (int i = 0; i < m_ConfusionMatrix.length; i++) {
		      if (i != classIndex) {
		        for (int j = 0; j < m_ConfusionMatrix.length; j++) {
		          if (j == classIndex) {
		            incorrect += m_ConfusionMatrix[i][j];
		          }
		          total += m_ConfusionMatrix[i][j];
		        }
		      }
		    }
		    if (total == 0) {
		      return 0;
		    }
		    return incorrect / total;
		  }
	  
	  public double recall(int classIndex) {

		    return truePositiveRate(classIndex);
	  }
	  
	  public double fMeasure(int classIndex) {

		    double precision = precision(classIndex);
		    double recall = recall(classIndex);
		    if ((precision + recall) == 0) {
		      return 0;
		    }
		    return 2 * precision * recall / (precision + recall);
		  }
	  
	  public double matthewsCorrelationCoefficient(int classIndex) {
		    double numTP = numTruePositives(classIndex);
		    double numTN = numTrueNegatives(classIndex);
		    double numFP = numFalsePositives(classIndex);
		    double numFN = numFalseNegatives(classIndex);
		    double n = (numTP * numTN) - (numFP * numFN);
		    double d = (numTP + numFP) * (numTP + numFN) * (numTN + numFP)
		        * (numTN + numFN);
		    d = Math.sqrt(d);

		    return n / d;
		  }
	  
	  public double numTruePositives(int classIndex) {

		    double correct = 0;
		    for (int j = 0; j < m_NumClasses; j++) {
		      if (j == classIndex) {
		        correct += m_ConfusionMatrix[classIndex][j];
		      }
		    }
		    return correct;
		  }
	  
	  public double numTrueNegatives(int classIndex) {

		    double correct = 0;
		    for (int i = 0; i < m_NumClasses; i++) {
		      if (i != classIndex) {
		        for (int j = 0; j < m_NumClasses; j++) {
		          if (j != classIndex) {
		            correct += m_ConfusionMatrix[i][j];
		          }
		        }
		      }
		    }
		    return correct;
		  }
	  
	  public double numFalsePositives(int classIndex) {

		    double incorrect = 0;
		    for (int i = 0; i < m_NumClasses; i++) {
		      if (i != classIndex) {
		        for (int j = 0; j < m_NumClasses; j++) {
		          if (j == classIndex) {
		            incorrect += m_ConfusionMatrix[i][j];
		          }
		        }
		      }
		    }
		    return incorrect;
		  }
	  
	  public double numFalseNegatives(int classIndex) {

		    double incorrect = 0;
		    for (int i = 0; i < m_NumClasses; i++) {
		      if (i == classIndex) {
		        for (int j = 0; j < m_NumClasses; j++) {
		          if (j != classIndex) {
		            incorrect += m_ConfusionMatrix[i][j];
		          }
		        }
		      }
		    }
		    return incorrect;
		  }
	  
	  
	  public double areaUnderROC(int classIndex) {

		    // Check if any predictions have been collected
		    if (m_Predictions == null) {
		      return Utils.missingValue();
		    } else {
		      ThresholdCurve tc = new ThresholdCurve();
		      Instances result = tc.getCurve(m_Predictions, classIndex);
		      return ThresholdCurve.getROCArea(result);
		    }
		  }
	  
	public String printConfusionMatrix() {
		    StringBuffer text = new StringBuffer();
		    char[] IDChars = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
		        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
		        'z' };
		    int IDWidth;
		    boolean fractional = false;

//		    if (!m_ClassIsNominal) {
//		      throw new Exception("Evaluation: No confusion matrix possible!");
//		    }

		    // Find the maximum value in the matrix
		    // and check for fractional display requirement
		    double maxval = 0;
		    for (int i = 0; i < m_NumClasses; i++) {
		      for (int j = 0; j < m_NumClasses; j++) {
		        double current = m_ConfusionMatrix[i][j];
		        if (current < 0) {
		          current *= -10;
		        }
		        if (current > maxval) {
		          maxval = current;
		        }
		        double fract = current - Math.rint(current);
		        if (!fractional && ((Math.log(fract) / Math.log(10)) >= -2)) {
		          fractional = true;
		        }
		      }
		    }

		    IDWidth = 1 + Math.max(
		        (int) (Math.log(maxval) / Math.log(10) + (fractional ? 3 : 0)),
		        (int) (Math.log(m_NumClasses) / Math.log(IDChars.length)));
		    text.append("=== Confusion Matrix ===\n").append("\n");
		    for (int i = 0; i < m_NumClasses; i++) {
		      if (fractional) {
		        text.append(" ").append(num2ShortID(i, IDChars, IDWidth - 3))
		            .append("   ");
		      } else {
		        text.append(" ").append(num2ShortID(i, IDChars, IDWidth));
		      }
		    }
		    text.append("   <-- classified as\n");
		    for (int i = 0; i < m_NumClasses; i++) {
		      for (int j = 0; j < m_NumClasses; j++) {
		        text.append(" ").append(
		            Utils.doubleToString(m_ConfusionMatrix[i][j], IDWidth,
		                (fractional ? 2 : 0)));
		      }
		      text.append(" | ").append(num2ShortID(i, IDChars, IDWidth)).append(" = ")
		          .append(m_ClassNames[i]).append("\n");
		    }
		    return text.toString();
		  }

	  protected String num2ShortID(int num, char[] IDChars, int IDWidth) {

		    char ID[] = new char[IDWidth];
		    int i;

		    for (i = IDWidth - 1; i >= 0; i--) {
		      ID[i] = IDChars[num % IDChars.length];
		      num = num / IDChars.length - 1;
		      if (num < 0) {
		        break;
		      }
		    }
		    for (i--; i >= 0; i--) {
		      ID[i] = ' ';
		    }

		    return new String(ID);
		  }
	
	  public void updateNumericScores(double[] predicted, double d,
		      double weight) {

		  double[] actual = new double[m_NumClasses];
		  actual[(int) d] = 1.0;
		  double diff;
		  double sumErr = 0, sumAbsErr = 0, sumSqrErr = 0;
		  double sumPriorAbsErr = 0, sumPriorSqrErr = 0;
		  for (int i = 0; i < m_NumClasses; i++) {
			  diff = predicted[i] - actual[i];
		      sumErr += diff;
		      sumAbsErr += Math.abs(diff);
		      sumSqrErr += diff * diff;
		      diff = (m_ClassPriors[i] / m_ClassPriorsSum) - actual[i];
		      sumPriorAbsErr += Math.abs(diff);
		      sumPriorSqrErr += diff * diff;
		  }
		  m_SumErr += weight * sumErr / m_NumClasses;
		  m_SumAbsErr += weight * sumAbsErr / m_NumClasses;
		  m_SumSqrErr += weight * sumSqrErr / m_NumClasses;
		  m_SumPriorAbsErr += weight * sumPriorAbsErr / m_NumClasses;
		  m_SumPriorSqrErr += weight * sumPriorSqrErr / m_NumClasses;
		}
	  
	  public final double meanAbsoluteError() {

		    return m_SumAbsErr / total_instances;
		  }
	  
	  public final double rootMeanSquaredError() {

		    return Math.sqrt(m_SumSqrErr / total_instances);
		  }
	  
	  public final double relativeAbsoluteError() throws Exception {
		    return 100 * meanAbsoluteError() / meanPriorAbsoluteError();
		  }
	  
	  public final double meanPriorAbsoluteError() {
		    return m_SumPriorAbsErr / total_instances;
		  }
	  
	  public final double rootRelativeSquaredError() {

		    return 100.0 * rootMeanSquaredError() / rootMeanPriorSquaredError();
		  }
	  
	  public final double rootMeanPriorSquaredError() {

		    return Math.sqrt(m_SumPriorSqrErr / total_instances);
		  }
	  
	  public double areaUnderPRC(int classIndex) {
		    // Check if any predictions have been collected
		    if (m_Predictions == null) {
		      return Utils.missingValue();
		    } else {
		      ThresholdCurve tc = new ThresholdCurve();
		      Instances result = tc.getCurve(m_Predictions, classIndex);
		      return ThresholdCurve.getPRCArea(result);
		    }
		  }
	  
	  public double weightedTruePositiveRate() {
		    double[] classCounts = new double[m_NumClasses];
		    double classCountSum = 0;

		    for (int i = 0; i < m_NumClasses; i++) {
		      for (int j = 0; j < m_NumClasses; j++) {
		        classCounts[i] += m_ConfusionMatrix[i][j];
		      }
		      classCountSum += classCounts[i];
		    }

		    double truePosTotal = 0;
		    for (int i = 0; i < m_NumClasses; i++) {
		      double temp = truePositiveRate(i);
		      truePosTotal += (temp * classCounts[i]);
		    }

		    return truePosTotal / classCountSum;
		  }
	  
	  public double weightedFalsePositiveRate() {
		    double[] classCounts = new double[m_NumClasses];
		    double classCountSum = 0;

		    for (int i = 0; i < m_NumClasses; i++) {
		      for (int j = 0; j < m_NumClasses; j++) {
		        classCounts[i] += m_ConfusionMatrix[i][j];
		      }
		      classCountSum += classCounts[i];
		    }

		    double falsePosTotal = 0;
		    for (int i = 0; i < m_NumClasses; i++) {
		      double temp = falsePositiveRate(i);
		      falsePosTotal += (temp * classCounts[i]);
		    }

		    return falsePosTotal / classCountSum;
		  }
	  
	  public double weightedPrecision() {
		    double[] classCounts = new double[m_NumClasses];
		    double classCountSum = 0;

		    for (int i = 0; i < m_NumClasses; i++) {
		      for (int j = 0; j < m_NumClasses; j++) {
		        classCounts[i] += m_ConfusionMatrix[i][j];
		      }
		      classCountSum += classCounts[i];
		    }

		    double precisionTotal = 0;
		    for (int i = 0; i < m_NumClasses; i++) {
		      double temp = precision(i);
		      precisionTotal += (temp * classCounts[i]);
		    }

		    return precisionTotal / classCountSum;
		  }
	  
	  public double weightedRecall() {
		    return weightedTruePositiveRate();
		  }
	  
	  public double weightedFMeasure() {
		    double[] classCounts = new double[m_NumClasses];
		    double classCountSum = 0;

		    for (int i = 0; i < m_NumClasses; i++) {
		      for (int j = 0; j < m_NumClasses; j++) {
		        classCounts[i] += m_ConfusionMatrix[i][j];
		      }
		      classCountSum += classCounts[i];
		    }

		    double fMeasureTotal = 0;
		    for (int i = 0; i < m_NumClasses; i++) {
		      double temp = fMeasure(i);
		      fMeasureTotal += (temp * classCounts[i]);
		    }

		    return fMeasureTotal / classCountSum;
		  }
	  
	  public double weightedMatthewsCorrelation() {
		    double[] classCounts = new double[m_NumClasses];
		    double classCountSum = 0;

		    for (int i = 0; i < m_NumClasses; i++) {
		      for (int j = 0; j < m_NumClasses; j++) {
		        classCounts[i] += m_ConfusionMatrix[i][j];
		      }
		      classCountSum += classCounts[i];
		    }

		    double mccTotal = 0;
		    for (int i = 0; i < m_NumClasses; i++) {
		      double temp = matthewsCorrelationCoefficient(i);
		      if (!Utils.isMissingValue(temp)) {
		        mccTotal += (temp * classCounts[i]);
		      }
		    }

		    return mccTotal / classCountSum;
		  }
	  
	  public double weightedAreaUnderROC() {
		    double[] classCounts = new double[m_NumClasses];
		    double classCountSum = 0;

		    for (int i = 0; i < m_NumClasses; i++) {
		      for (int j = 0; j < m_NumClasses; j++) {
		        classCounts[i] += m_ConfusionMatrix[i][j];
		      }
		      classCountSum += classCounts[i];
		    }

		    double aucTotal = 0;
		    for (int i = 0; i < m_NumClasses; i++) {
		      double temp = areaUnderROC(i);
		      if (!Utils.isMissingValue(temp)) {
		        aucTotal += (temp * classCounts[i]);
		      }
		    }

		    return aucTotal / classCountSum;
		  }
	  
	  public double weightedAreaUnderPRC() {
		    double[] classCounts = new double[m_NumClasses];
		    double classCountSum = 0;

		    for (int i = 0; i < m_NumClasses; i++) {
		      for (int j = 0; j < m_NumClasses; j++) {
		        classCounts[i] += m_ConfusionMatrix[i][j];
		      }
		      classCountSum += classCounts[i];
		    }

		    double auprcTotal = 0;
		    for (int i = 0; i < m_NumClasses; i++) {
		      double temp = areaUnderPRC(i);
		      if (!Utils.isMissingValue(temp)) {
		        auprcTotal += (temp * classCounts[i]);
		      }
		    }

		    return auprcTotal / classCountSum;
		  }
	  
	  public String toClassDetailsString() throws Exception {

		    StringBuffer text = new StringBuffer("=== Detailed Accuracy By Class ===\n"+ 
		    		"\n                 TP Rate  FP Rate" + "  Precision  Recall"
		        + "  F-Measure  MCC    ROC Area  PRC Area  Class\n");
		    for (int i = 0; i < m_NumClasses; i++) {
		      text.append(
		          "               " + Utils.doubleToString(truePositiveRate(i), 7, 3))
		          .append("  ");
		      text.append(Utils.doubleToString(falsePositiveRate(i), 7, 3))
		          .append("  ");
		      text.append(Utils.doubleToString(precision(i), 7, 3)).append("    ");
		      text.append(Utils.doubleToString(recall(i), 7, 3)).append(" ");
		      text.append(Utils.doubleToString(fMeasure(i), 7, 3)).append("    ");
		      double mat = matthewsCorrelationCoefficient(i);
		      if (Utils.isMissingValue(mat)) {
		        text.append("  ?  ").append("   ");
		      } else {
		        text.append(
		            Utils.doubleToString(matthewsCorrelationCoefficient(i), 7, 3))
		            .append("");
		      }

		      double rocVal = areaUnderROC(i);
		      if (Utils.isMissingValue(rocVal)) {
		        text.append(" ?    ").append("   ");
		      } else {
		        text.append(Utils.doubleToString(rocVal, 7, 3)).append("   ");
		      }
		      double prcVal = areaUnderPRC(i);
		      if (Utils.isMissingValue(prcVal)) {
		        text.append("  ?    ").append("     ");
		      } else {
		        text.append(Utils.doubleToString(prcVal, 7, 3)).append("     ");
		      }

		      text.append(m_ClassNames[i]).append('\n');
		    }

		    text.append("Weighted Avg.  "
		        + Utils.doubleToString(weightedTruePositiveRate(), 7, 3));
		    text.append("  " + Utils.doubleToString(weightedFalsePositiveRate(), 7, 3));
		    text.append("  " + Utils.doubleToString(weightedPrecision(), 7, 3));
		    text.append("    " + Utils.doubleToString(weightedRecall(), 7, 3));
		    text.append(" " + Utils.doubleToString(weightedFMeasure(), 7, 3));
		    text.append("    "
		        + Utils.doubleToString(weightedMatthewsCorrelation(), 7, 3));
		    text.append("" + Utils.doubleToString(weightedAreaUnderROC(), 7, 3));
		    text.append("   " + Utils.doubleToString(weightedAreaUnderPRC(), 7, 3));
		    text.append("\n");

		    return text.toString();
		  }
	  
	  
	  
	  public void setPredictions(Instance instance, double[] dist){
		  if (m_Predictions == null)
			  m_Predictions = new FastVector();
	      m_Predictions.addElement(new NominalPrediction(instance.classValue(),dist, instance.weight()));
	  }
	  
	public String toString(){
        output.append("Correctly Classified Instances     ");
        output.append(Utils.doubleToString(correct(), 12, 4) + "     "
            + Utils.doubleToString(pctCorrect(), 12, 4) + " %\n");
        output.append("Incorrectly Classified Instances   ");
        output.append(Utils.doubleToString(incorrect(), 12, 4) + "     "
            + Utils.doubleToString(pctIncorrect(), 12, 4) + " %\n");
        output.append("Kappa statistic                    ");
        output.append(Utils.doubleToString(kappa(), 12, 4) + "\n");
        output.append("Mean absolute error                ");
        output.append(Utils.doubleToString(meanAbsoluteError(), 12, 4) + "\n");
        output.append("Root mean squared error            ");
        output.append(Utils.doubleToString(rootMeanSquaredError(), 12, 4) + "\n");
        output.append("Relative absolute error            ");
        try {
			output.append(Utils.doubleToString(relativeAbsoluteError(), 12, 4)
			    + " %\n");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        output.append("Root relative squared error        ");
        output.append(Utils.doubleToString(rootRelativeSquaredError(), 12, 4)
            + " %\n");
        try {
			output.append(toClassDetailsString());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		output.append(printConfusionMatrix());
		return output.toString();
	}
}
