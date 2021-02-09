/*
 * Author: Tommaso Apicella
 * Email: tommaso.apicella@edu.unige.it
 */

package com.example.polarityclassificationusingsaliency.utils;

import com.example.polarityclassificationusingsaliency.classes.PolarityClassifier;
import com.example.polarityclassificationusingsaliency.classes.Recognition;

import static java.lang.Math.abs;

public class AlgoUtils {
    public static int retrieveIndex(float[] array) {
        if (array == null || array.length == 0) return -1; // null or empty

        int largest = 0;
        for (int i = 1; i < array.length; i++) {
            if (abs(array[i]) > abs(array[largest])) largest = i;
        }
        return largest; // position of the first largest found
    }

    public static void fillBin(PolarityClassifier polClassPatches, float[] binScore, Recognition polRes, int originalImageArea, int patchArea) {
        // Select correct sign
        float conf = 0.0f;
        if (polRes.getTitle().equals(polClassPatches.getLabels().get(0))) {
            conf = (-1) * polRes.getConfidence();
        } else {
            conf = polRes.getConfidence();
        }
        // Compute overlapping percentage
        float ovPerc = (float) patchArea / originalImageArea;
        // Select correct bin
        for (int i = 0; i < binScore.length; i++) {
            if (ovPerc < ((i + 1) * (1.0 / binScore.length))) {
                binScore[i] += conf;
                break;
            }
        }
    }
}
