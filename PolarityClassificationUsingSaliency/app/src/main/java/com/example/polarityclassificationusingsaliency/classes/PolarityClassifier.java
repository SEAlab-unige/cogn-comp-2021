/*
 * Author: Tommaso Apicella
 * Email: tommaso.apicella@edu.unige.it
 */

package com.example.polarityclassificationusingsaliency.classes;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;


public class PolarityClassifier {
    // New type to encode the type of tensor's data
    private enum TensorType {
        NONE, UINT8, FLOAT32 // in Java there is no FLOAT16
    }

    // Constants
    private static final String TAG = "PolarityClassifier";
    private static final int NUM_CHANNELS = 3;

    // Variables
    private int imageSizeX; // classifier input tensor width
    private int imageSizeY; // classifier input tensor height
    private int numChannels; // number of images' channels
    private String modelPath; // model's path
    private String labelsPath; // labels' path
    private Interpreter tflite; // tflite interpreter
    private List<String> labels; // list containing the labels
    private TensorType inputType; // type of input tensor
    private TensorType outputType; // type of output tensor

    // Default constructor
    public PolarityClassifier() {
        this.imageSizeX = 0;
        this.imageSizeY = 0;
        this.modelPath = "";
        this.labelsPath = "";
        this.inputType = TensorType.NONE;
        this.outputType = TensorType.NONE;
    }

    // Close the interpreter and model to release resources
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
    }

    // Getters
    public String getModelPath() {
        return modelPath;
    }

    public String getLabelsPath() {
        return labelsPath;
    }

    public List<String> getLabels() {
        return labels;
    }

    // Setters
    public void setParams(String modelPath, String labelsPath, Activity activity) {
        // Set model's path
        setModelPath(modelPath);
        // Set labels' path
        setLabelsPath(labelsPath);
        // Load classifier model
        try {
            tflite = new Interpreter(loadModelFile(activity, getModelPath()));
        } catch (IOException e) {
            e.printStackTrace();
            Log.i(TAG, "PolarityClassifier model file not found!");
        }
        // Loads labels out from the label file.
        try {
            labels = FileUtil.loadLabels(activity, getLabelsPath());
        } catch (IOException e) {
            e.printStackTrace();
            // Display TOAST
            Log.i(TAG, "Labels file not found!");
        }
        // Read shape of input tensor
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape();
        if (imageShape.length == 4) {  // {batch, width, height, 3}
            imageSizeX = imageShape[1];
            imageSizeY = imageShape[2];
        } else if (imageShape.length == 3) { // {width, height, 3}
            imageSizeX = imageShape[0];
            imageSizeY = imageShape[1];
        }
        // Read type of input tensor
        DataType inputDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        if (inputDataType == DataType.UINT8) {
            inputType = TensorType.UINT8;
        } else if (inputDataType == DataType.FLOAT32) {
            inputType = TensorType.FLOAT32;
        }
        // Read type of output tensor
        DataType outputDataType = tflite.getOutputTensor(imageTensorIndex).dataType();
        if (outputDataType == DataType.UINT8) {
            outputType = TensorType.UINT8; // Full integer quantization
        } else if (outputDataType == DataType.FLOAT32) {
            outputType = TensorType.FLOAT32;
        }
        // Set channels
        setNumChannels(NUM_CHANNELS); // RGB images
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
    }

    public void setLabelsPath(String labelsPath) {
        this.labelsPath = labelsPath;
    }

    public void setNumChannels(int numChannels) {
        this.numChannels = numChannels;
    }


    // Load classifier model method
    private MappedByteBuffer loadModelFile(Activity activity, String modelPath) throws IOException {
        AssetFileDescriptor assetFileDescriptor = activity.getAssets().openFd(modelPath);
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startoffset = assetFileDescriptor.getStartOffset();
        long declaredLength = assetFileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength);
    }

    // Inference method
    public Recognition polarityInference(Bitmap bitmap) {
        // Prepare input
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, imageSizeX, imageSizeY, false);
        ByteBuffer inputBuffer = convertBitmapToByteBuffer(scaledBitmap);
        // Prepare output
        float[][] labelProbArray = new float[1][labels.size()];
        if(outputType == TensorType.UINT8){
            byte[][] byteLabelProbArray = new byte[1][labels.size()];
            // Inference
            tflite.run(inputBuffer, byteLabelProbArray);
            // Map byte range into float
            for (int i = 0; i < labels.size(); i++) {
                labelProbArray[0][i] = getNormalizedProbability(i, byteLabelProbArray);
            }
        }
        else{
            // Inference
            tflite.run(inputBuffer, labelProbArray);
        }
        // Create a map with class and confidence
        Map<String, Float> labelProb = new HashMap<>();
        for (int i = 0; i < labels.size(); i++) {
            labelProb.put(labels.get(i), labelProbArray[0][i]);
        }
        // Retrieve actual prediction
        return retrievePrediction(labelProb);
    }

    // Convert Bitmap to ByteBuffer
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer imgData = null;
        if (this.inputType == TensorType.UINT8) {
            imgData = ByteBuffer.allocateDirect(numChannels * imageSizeX * imageSizeY);
        } else if (this.inputType == TensorType.FLOAT32) {
            imgData = ByteBuffer.allocateDirect(4 * numChannels * imageSizeX * imageSizeY); // FLOAT32 is 4 bytes
        }
        imgData.order(ByteOrder.nativeOrder());
        int[] intValues = new int[imageSizeX * imageSizeY];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        for (int i = 0; i < imageSizeX; ++i) {
            for (int j = 0; j < imageSizeY; ++j) {
                final int val = intValues[pixel++];
                if (this.inputType == TensorType.UINT8) {
                    imgData.put((byte) ((val >> 16) & 0xFF));
                    imgData.put((byte) ((val >> 8) & 0xFF));
                    imgData.put((byte) ((val) & 0xFF));
                } else if (this.inputType == TensorType.FLOAT32) {
                    final int IMAGE_MEAN = 128;
                    final float IMAGE_STD = 128.0f;
                    // Process image channels
                    imgData.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        return imgData;
    }

    // Retrieve the prediction
    private Recognition retrievePrediction(Map<String, Float> labelProb) {
        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        1,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
            pq.add(new Recognition(entry.getKey(), entry.getValue(), null));
        }

        return pq.peek();
    }

    // Normalize probability (quantized model case)
    float getNormalizedProbability(int labelIndex, byte[][] labelProbArray) {
        return (labelProbArray[0][labelIndex] & 0xff) / 255.0f;
    }
}
