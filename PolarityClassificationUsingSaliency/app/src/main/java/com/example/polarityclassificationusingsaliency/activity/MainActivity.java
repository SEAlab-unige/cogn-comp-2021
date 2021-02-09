/*
* Author: Tommaso Apicella
* Email: tommaso.apicella@edu.unige.it
 */

package com.example.polarityclassificationusingsaliency.activity;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.OpenableColumns;
import android.view.Gravity;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.example.polarityclassificationusingsaliency.classes.PolarityClassifier;
import com.example.polarityclassificationusingsaliency.R;
import com.example.polarityclassificationusingsaliency.classes.Recognition;
import com.example.polarityclassificationusingsaliency.classes.SaliencyDetector;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import static com.example.polarityclassificationusingsaliency.utils.AlgoUtils.fillBin;
import static com.example.polarityclassificationusingsaliency.utils.AlgoUtils.retrieveIndex;
import static com.example.polarityclassificationusingsaliency.utils.FileUtils.convertArrayToList;
import static com.example.polarityclassificationusingsaliency.utils.FileUtils.retrieveFileNames;
import static com.example.polarityclassificationusingsaliency.utils.ImageUtils.cropImage;
import static java.lang.Math.abs;
import static java.lang.Math.round;

public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {
    // Constants
    private static final String TAG = "MainActivity";
    private static final String IMAGES_DIR = "test_set";
    private static final String POLARITY_CLAS_DIR = "models/polarity_classifier";
    private static final String POLARITY_CLAS_ENTIRE_IMG = POLARITY_CLAS_DIR + "/entire_image";
    private static final String POLARITY_CLAS_PATCHES = POLARITY_CLAS_DIR + "/patches";
    private static final String POLARITY_LABELS_FILE = "labels.txt";
    private static final String SALIENCY_DET_DIR = "models/saliency_detector";
    private static final String EMPTY_TEXT = "                      ";
    private static final int NUM_BINS = 5;
    private static final int PICK_IMAGE = 11;

    // Widgets' variables
    Button btn_loadDataset;
    Button btn_loadDatasetFromStorage;
    Spinner spn_selPolClass;
    Spinner spn_selSalDet;
    Button btn_inference;
    TextView txtv_time;
    TextView txtv_result;
    TextView txtv_confidence;

    // Private variables
    private boolean testSetSelected;
    private boolean polClassSelected;
    private boolean salDetSelected;
    private boolean loadfromAsset;
    private ArrayList<String> images_names;
    private PolarityClassifier polClassEntireImage;
    private PolarityClassifier polClassPatches;
    private SaliencyDetector salDet;
    private Uri selectedImageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Link variables to XML widgets
        btn_loadDataset = (Button) findViewById(R.id.loadTestSetButton);
        btn_loadDatasetFromStorage = (Button) findViewById(R.id.loadTestFromStorageButton);
        spn_selPolClass = (Spinner) findViewById(R.id.selPolClassSpinner);
        spn_selSalDet = (Spinner) findViewById(R.id.selSalDetSpinner);
        btn_inference = (Button) findViewById(R.id.inferenceButton);
        txtv_time = (TextView) findViewById(R.id.inferenceTimeTextView);
        txtv_result = (TextView) findViewById(R.id.inferenceResultTextView);
        txtv_confidence = (TextView) findViewById(R.id.inferenceConfidenceTextView);

        // Load categories in spinner
        setUpPolaritySpinner(spn_selPolClass, POLARITY_CLAS_DIR + "/format.txt");
        setUpSaliencySpinner(spn_selSalDet, SALIENCY_DET_DIR);

        // Instantiate polarity classifier
        polClassEntireImage = new PolarityClassifier();
        polClassPatches = new PolarityClassifier();

        // Instantiate saliency detector
        salDet = new SaliencyDetector();

        // Initialise variables
        testSetSelected = false;
        polClassSelected = false;
        salDetSelected = false;
        loadfromAsset = false;
        images_names = new ArrayList<String>();
    }

    public void retrieveImages(View v) {
        // Clear images list
        images_names.clear();
        try {
            images_names = convertArrayToList(IMAGES_DIR, retrieveFileNames(getApplicationContext(), IMAGES_DIR));
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Display TOAST
        String text = "Found " + images_names.size() + " image!!";
        displayToast(text);

        // Reset GUI
        resetResults();

        testSetSelected = true;
        loadfromAsset = true;
    }

    public void retrieveImageFromStorage(View v) {
        // Create intent to pick image
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("*/*");
        startActivityForResult(intent, PICK_IMAGE);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE && resultCode == Activity.RESULT_OK) {
            if (data == null) {
                // Display TOAST
                String text = "Image loading failed!!";
                displayToast(text);
                testSetSelected = false;
                return;
            }
            // Reset GUI
            images_names.clear();
            selectedImageUri = data.getData();

            // Get image name and add it to the list
            Cursor returnCursor =
                    getContentResolver().query(selectedImageUri, null, null, null, null);
            int nameIndex = returnCursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
            returnCursor.moveToFirst();
            String imageName = returnCursor.getString(nameIndex);
            images_names.add(imageName);

            // Display TOAST
            String text = images_names.size() + " image successfully loaded!!";
            displayToast(text);
            resetResults();
            testSetSelected = true;
            loadfromAsset = false;
        }
    }

    private void setUpPolaritySpinner(Spinner spn_sel, String formatPath) {
        // Spinner drop down elements
        List<String> categories = new ArrayList<String>();
        categories.add(EMPTY_TEXT); // load at least empty category
        try (BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(formatPath)))) {
            String line;
            while ((line = br.readLine()) != null) {
                categories.add(line);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        // Creating adapter for spinner
        ArrayAdapter<String> dataAdapter = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, categories);
        // Drop down layout style - list view with radio button
        dataAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        // Attaching data adapter to spinner
        spn_sel.setAdapter(dataAdapter);
        // Attaching listener
        spn_sel.setOnItemSelectedListener(this);
    }

    private void setUpSaliencySpinner(Spinner spn_sel, String model_path) {
        // Spinner drop down elements
        List<String> categories = new ArrayList<String>();
        categories.add(EMPTY_TEXT); // load at least empty category
        // Retrieve files in model_path
        try {
            String[] files_names = retrieveFileNames(getApplicationContext(), model_path);
            // Update categories
            for (String s : files_names) {
                if (s.contains(".tflite")) {
                    String spin_choice = s.substring(0, s.lastIndexOf("."));
                    categories.add(spin_choice);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        // Creating adapter for spinner
        ArrayAdapter<String> dataAdapter = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, categories);
        // Drop down layout style - list view with radio button
        dataAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        // Attaching data adapter to spinner
        spn_sel.setAdapter(dataAdapter);
        // Attaching listener
        spn_sel.setOnItemSelectedListener(this);
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        // On selecting a spinner item
        String item = parent.getItemAtPosition(position).toString();
        if (item != EMPTY_TEXT) {
            if (parent == spn_selPolClass) {
                polClassEntireImage.setParams(selectClassifier(POLARITY_CLAS_ENTIRE_IMG, item), (POLARITY_CLAS_DIR + "/" + POLARITY_LABELS_FILE), this);
                polClassPatches.setParams(selectClassifier(POLARITY_CLAS_PATCHES, item), (POLARITY_CLAS_DIR + "/" + POLARITY_LABELS_FILE), this);
                resetResults();
                polClassSelected = true;
            } else if (parent == spn_selSalDet) {
                salDet.setParams((SALIENCY_DET_DIR + "/" + item + ".tflite"), this);
                resetResults();
                salDetSelected = true;
            }
            // Display TOAST
            String text = "Selected: " + item;
            displayToast(text);
        } else {
            if (parent == spn_selPolClass) {
                polClassSelected = false;
            } else if (parent == spn_selSalDet) {
                salDetSelected = false;
            }
            resetResults();
        }
    }

    private String selectClassifier(String modelsDir, String modelName) {
        String filePath = null;
        try {
            String[] files_names = retrieveFileNames(getApplicationContext(), modelsDir);
            for (String f : files_names) {
                if (f.contains(modelName)) {
                    filePath = modelsDir + "/" + f;
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return filePath;
    }

    private void resetResults() {
        // Reset results widgets
        txtv_time.setText("___");
        txtv_result.setText("___");
        txtv_confidence.setText("___");
    }

    @Override
    public void onNothingSelected(AdapterView<?> arg0) {
        // TODO Auto-generated method stub
    }
    
    public void runInference(View v) {
        // At least images and polarity classifier must be selected
        if (!(testSetSelected && polClassSelected)) {
            // Display TOAST
            String text = "Test set or polarity model missing!";
            displayToast(text);
            return;
        }
        long startTime = 0l;
        long lastProcessingTimeMs = 0l;
        // Retrieve initial time
        startTime = SystemClock.uptimeMillis();
        // For each image run inference
        for (String im : images_names) {
            // Convert image to bitmap
            InputStream inputStream = null;
            try {
                if (loadfromAsset) {
                    Context context = getApplicationContext();
                    inputStream = context.getAssets().open(im);
                } else {
                    inputStream = getContentResolver().openInputStream(selectedImageUri);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);

            // -------------- SALIENCY DETECTION --------------
            List<Bitmap> inputBitmapList = new ArrayList<Bitmap>();
            if (salDetSelected) {
                // Retrieve salient patches
                List<Recognition> saliencyResults = salDet.saliencyDetection(bitmap);
                if (!saliencyResults.isEmpty()) {
                    // For each patch
                    for (Recognition patch : saliencyResults) {
                        // Crop image
                        Bitmap inputBitmap = cropImage(bitmap, patch.getLocation());
                        inputBitmapList.add(inputBitmap);
                    }
                }
            }

            // -------------- POLARITY CLASSIFICATION --------------
            Recognition tempPred = polClassEntireImage.polarityInference(bitmap);
            Recognition pred;
            if (salDetSelected && !inputBitmapList.isEmpty()) {
                int originalImageArea = bitmap.getWidth() * bitmap.getHeight();
                // Create bins
                float[] binScore = new float[NUM_BINS];
                for (int i = 0; i < inputBitmapList.size(); i++) {
                    Bitmap b = inputBitmapList.get(i);
                    // Compute area of the patch
                    int patchArea = b.getWidth() * b.getHeight();
                    // Classify patch
                    Recognition polRes = polClassPatches.polarityInference(b);
                    // Fill correct bin
                    fillBin(polClassPatches, binScore, polRes, originalImageArea, patchArea);
                }
                // Get index of the maximum score (absolute value)
                int indexMax = retrieveIndex(binScore);
                if (abs(binScore[indexMax]) > tempPred.getConfidence()) {
                    if (binScore[indexMax] < 0.0f) {
                        pred = new Recognition(polClassPatches.getLabels().get(0), -10.0f);
                    } else {
                        pred = new Recognition(polClassPatches.getLabels().get(1), -10.0f);
                    }
                } else {
                    pred = tempPred;
                }
            } else {
                pred = tempPred;
            }
            // Retrieve inference time
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            // Display inference time
            txtv_time.setText(String.valueOf(lastProcessingTimeMs));
            // Display inference result
            txtv_result.setText(pred.getTitle());
            // Display inference confidence
            if (pred.getConfidence() != -10.0f) {
                txtv_confidence.setText(String.valueOf(round(pred.getConfidence() * 1000) / 10.0));
            } else {
                txtv_confidence.setText("---");
            }
        }
    }

    public void displayToast(String text) {
        // Display TOAST
        Context context = getApplicationContext();
        Toast t = Toast.makeText(context, text, Toast.LENGTH_SHORT);
        t.setGravity(Gravity.BOTTOM, 0, 0);
        t.show();
    }
}