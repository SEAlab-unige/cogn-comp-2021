/*
 * Author: Tommaso Apicella
 * Email: tommaso.apicella@edu.unige.it
 */

package com.example.polarityclassificationusingsaliency.classes;

public class Recognition {

    //Display name for the recognition.
    private final String title;

    // A sortable score for how good the recognition is relative to others. Higher should be better.
    private final Float confidence;

    // Boxes coordinates (if any)
    private final int[] location;

    // Constructors
    public Recognition(final String title, final Float confidence) {
        this.title = title;
        this.confidence = confidence;
        this.location = null;
    }

    public Recognition(final String title, final Float confidence, final int[] location) {
        this.title = title;
        this.confidence = confidence;
        this.location = location;
    }

    // Getters
    public int[] getLocation() {
        return location;
    }

    public String getTitle() {
        return title;
    }

    public Float getConfidence() {
        return confidence;
    }

    // Convert to string
    @Override
    public String toString() {
        String resultString = "";

        if (title != null) {
            resultString += title + " ";
        }

        if (confidence != null) {
            resultString += String.format("(%.1f%%) ", confidence * 100.0f);
        }

        if (location != null) {
            resultString += location + " ";
        }

        return resultString.trim();
    }
}
