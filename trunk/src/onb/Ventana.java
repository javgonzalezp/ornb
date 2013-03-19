package onb;

public class Ventana {
	double tamaño, total=0, aciertos=0, errores=0;
	double probAciertos, probErrores;
	String[] classes;
	int[] entropia;
	boolean referencia = false;
	
	public Ventana(double tamVentana, boolean referencia, String[] classes){
		this.tamaño = (int) tamVentana;
		this.referencia = referencia;
		this.classes = classes;
		entropia = new int[classes.length];
	}
	
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
	
	public void updateProbs(){
		probAciertos=aciertos/total;
		probErrores=errores/total;
	}
	
	public void addCount(){
		total++;
	}
	
	public void resetCounters(){
		aciertos=0;
		errores=0;
		total=0;
	}
	
	public double getProbAciertos(){
		return probAciertos;
	}
	
	public double getProbErrores(){
		return probErrores;
	}

	public double getEntropia() {
//		double sum = 0;
//		for(int i=0; i<classes.length; i++){
//			double prob = entropia[i]/total;
//			double aux=-prob*(Math.log(prob)/Math.log(2));
//			sum += aux;
//		}
		double aux1=-probAciertos*(Math.log(probAciertos)/Math.log(2));
		double aux2=-probErrores*(Math.log(probErrores)/Math.log(2));
		if(probAciertos==0)
			aux1=0;
		if(probErrores==0)
			aux2=0;
		return aux1+aux2;
	}

	public void addClass(String stringValue) {
		if(referencia){
			if(total<tamaño){
				total++;
				for(int i=0; i<classes.length; i++){
					if(classes[i].equalsIgnoreCase(stringValue))
						entropia[i]++;
				}
			}
		}
		else{
			total++;
			for(int i=0; i<classes.length; i++){
				if(classes[i].equalsIgnoreCase(stringValue))
					entropia[i]++;
			}
		}
			

	}
}
