package com.bodo.iceberg;

import com.bodo.iceberg.gson.InterfaceAdapter;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.iceberg.BlobMetadata;
import org.apache.iceberg.GenericBlobMetadata;
import org.apache.iceberg.GenericStatisticsFile;
import org.apache.iceberg.StatisticsFile;

public class BodoStatisticFile {
  public static StatisticsFile fromJson(String infoStr) {
    Gson gson =
        new GsonBuilder()
            .registerTypeAdapter(
                BlobMetadata.class, new InterfaceAdapter<BlobMetadata>(GenericBlobMetadata.class))
            .create();
    return gson.fromJson(infoStr, GenericStatisticsFile.class);
  }
}
