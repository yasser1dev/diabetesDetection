import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class MainModel {
    public static void main(String[] args) throws IOException, InterruptedException {
        double learningRate=0.001;
        int inputNum=8;
        int hiddenNeuroneNumLayer1=8;
        int hiddenNeuroneNumLayer2=5;


        int outputSize=2;
        System.out.println("Model creation");
        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder().
                seed(123)
                .updater(new Adam(learningRate))
                //.weightInit(WeightInit.XAVIER) par default Xavier est choisi
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(inputNum).nOut(hiddenNeuroneNumLayer1).activation(Activation.SIGMOID).build())
                .layer(1,new DenseLayer.Builder()
                        .nIn(hiddenNeuroneNumLayer1).nOut(hiddenNeuroneNumLayer2).activation(Activation.SIGMOID).build())
                .layer(2,new OutputLayer.Builder()
                        .nIn(hiddenNeuroneNumLayer2).nOut(outputSize).lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .activation(Activation.SOFTMAX).build())
                .build();

        MultiLayerNetwork net=new MultiLayerNetwork(configuration);
        net.init();

        System.out.println("web server starting .......");
        UIServer uiServer=UIServer.getInstance();
        InMemoryStatsStorage inMemoryStatsStorage=new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);

        net.setListeners(new StatsListener(inMemoryStatsStorage));
        int nEpoch=250;  //number of repeating the training
        int batchSize=1; //number of rows to feed the model and do error optimization
        int classIndex=8;
        System.out.println("Model training");

        File trainData=new ClassPathResource("train_data.csv").getFile();
        RecordReader recordReader=new CSVRecordReader();
        recordReader.initialize(new FileSplit(trainData));

        DataSetIterator dataSetIteratorTrain=new RecordReaderDataSetIterator(recordReader,batchSize,classIndex,outputSize);

        while(dataSetIteratorTrain.hasNext()){
            System.out.println("--------------------------------");
            DataSet dataSet=dataSetIteratorTrain.next();
            System.out.println(dataSet.getFeatures());
            System.out.println(dataSet.getLabels());
        }



        for(int i=0;i<nEpoch;i++) {
            net.fit(dataSetIteratorTrain);


        }


        System.out.println("Model evalutaion");
        File testData=new ClassPathResource("test_data.csv").getFile();
        RecordReader recordReader1=new CSVRecordReader();
        recordReader1.initialize(new FileSplit(testData));
        DataSetIterator dataSetIteratorTest=new RecordReaderDataSetIterator(recordReader1,1,classIndex,outputSize);
        Evaluation evaluation=new Evaluation();
        while(dataSetIteratorTest.hasNext()) {
            org.nd4j.linalg.dataset.DataSet dataSetTest = dataSetIteratorTest.next();
            INDArray features = dataSetTest.getFeatures();
            INDArray target = dataSetTest.getLabels();
            INDArray predictedLabels = net.output(features);
            evaluation.eval(predictedLabels, target);
        }

        System.out.print(evaluation.stats());
        ModelSerializer.writeModel(net,"diabetes2.zip",true);

    }
}
