/*
 * Author: Tommaso Apicella
 * Email: tommaso.apicella@edu.unige.it
 */

package com.example.polarityclassificationusingsaliency.utils;

import android.graphics.Bitmap;

public class ImageUtils {
    public static Bitmap cropImage(Bitmap bitmap, int[] rect) {
        // rect is [xmin, ymin, xmax, ymax]
        return Bitmap.createBitmap(bitmap, rect[0], rect[1], (rect[2] - rect[0]), (rect[3] - rect[1]));
    }
}
