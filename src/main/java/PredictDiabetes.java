import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class PredictDiabetes {
    public static void main(String[] args) throws IOException {
        MultiLayerNetwork model= ModelSerializer.restoreMultiLayerNetwork("diabetes.zip");
        INDArray dataToPredict= Nd4j.create(new double[][]{
                {2	,56	,56	,28	,45	,24.2	,0.332	,22}
        });
        INDArray output=model.output(dataToPredict);
        String[] classes={"has not diabetes","has diabetes"};
        int[] outputIndex=output.argMax(1).toIntVector();
        System.out.println("Prediction : "+classes[outputIndex[0]]);
    }
}
