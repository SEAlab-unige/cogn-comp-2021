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

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.Math.round;


public class SaliencyDetector {
    // New type to encode the type of tensor's data
    private enum TensorType {
        NONE, UINT8, FLOAT32 // in Java there is no FLOAT16
    }

    // Constants
    private static final String TAG = "Saliency Detector";
    private static final int NUM_DETECTIONS = 10; // number of detections
    private static final int NUM_CHANNELS = 3;
    private static final float SAL_THRESHOLD = 0.3f;

    // Variables
    private int imageSizeX; // classifier input tensor width
    private int imageSizeY; // classifier input tensor height
    private int numChannels; // number of images' channels
    private String modelPath; // model's path
    private Interpreter tflite; // tflite interpreter
    private TensorType inputType; // type of input tensor
    private TensorType outputType; // type of output tensor

    // Default constructor
    public SaliencyDetector() {
        this.imageSizeX = 0;
        this.imageSizeY = 0;
        this.modelPath = "";
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

    // Setters
    public void setParams(String modelPath, Activity activity) {
        // Set model's path
        setModelPath(modelPath);
        // Load classifier model
        try {
            tflite = new Interpreter(loadModelFile(activity, getModelPath()));
        } catch (IOException e) {
            e.printStackTrace();
            Log.i(TAG, "Saliency model file not found!");
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
            outputType = TensorType.UINT8;
        } else if (outputDataType == DataType.FLOAT32) {
            outputType = TensorType.FLOAT32;
        }
        // Set channels
        setNumChannels(NUM_CHANNELS); // RGB images
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
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
    public List<Recognition> saliencyDetection(Bitmap bitmap) {
        // Prepare input
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, imageSizeX, imageSizeY, false);
        ByteBuffer inputBuffer = convertBitmapToByteBuffer(scaledBitmap);
        // Prepare output
        float[][][] outputLocations = new float[1][NUM_DETECTIONS][4];
        float[][] outputClasses = new float[1][NUM_DETECTIONS];
        float[][] outputScores = new float[1][NUM_DETECTIONS];
        float[] numDetections = new float[1];
        Object[] inputArray = {inputBuffer};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);

        // Run the inference call.
        tflite.runForMultipleInputsOutputs(inputArray, outputMap);

        int numDetectionsOutput = Math.min(NUM_DETECTIONS, (int) numDetections[0]); // cast from float to integer, use min for safety

        return retrievePredictions(outputLocations, numDetectionsOutput, outputScores, bitmap.getWidth(), bitmap.getHeight());
    }

    // Convert Bitmap to ByteBuffer method
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
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
                if (this.inputType == TensorType.UINT8) {  // Quantized model
                    imgData.put((byte) ((val >> 16) & 0xFF));
                    imgData.put((byte) ((val >> 8) & 0xFF));
                    imgData.put((byte) ((val) & 0xFF));
                } else if (this.inputType == TensorType.FLOAT32) { // Float model
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

    private List<Recognition> retrievePredictions(float[][][] outputLocations, int numDetectionsOutput, float[][] outputScores, int im_width, int im_height) {
        final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
        for (int i = 0; i < numDetectionsOutput; ++i) {
            if (outputScores[0][i] > SAL_THRESHOLD) {
                // Check xmin
                int xmin = 0;
                if (round(outputLocations[0][i][1] * im_width) > 0) {
                    xmin = round(outputLocations[0][i][1] * im_width);
                }
                // Check xmax
                int xmax = im_width;
                if (round(outputLocations[0][i][3] * im_width) < im_width) {
                    xmax = round(outputLocations[0][i][3] * im_width);
                }
                // Check ymin
                int ymin = 0;
                if (round(outputLocations[0][i][0] * im_height) > 0) {
                    ymin = round(outputLocations[0][i][0] * im_height);
                }
                // Check ymax
                int ymax = im_height;
                if (round(outputLocations[0][i][2] * im_height) < im_height) {
                    ymax = round(outputLocations[0][i][2] * im_height);
                }
                // Build detection
                final int[] detection =
                        new int[]{
                                xmin,
                                ymin,
                                xmax,
                                ymax};
                // Fill recognition list
                recognitions.add(
                        new Recognition(
                                "",
                                outputScores[0][i],
                                detection));
            }
        }
        return recognitions;
    }
}
