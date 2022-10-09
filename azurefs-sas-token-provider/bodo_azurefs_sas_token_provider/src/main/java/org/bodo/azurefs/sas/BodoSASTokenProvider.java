package org.bodo.azurefs.sas;

import java.io.FileInputStream;
import java.io.IOException;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.azurebfs.extensions.SASTokenProvider;
import org.apache.hadoop.security.AccessControlException;

public class BodoSASTokenProvider implements SASTokenProvider {

  public void initialize(Configuration configuration, String accountName) throws IOException {
    // Nop
  }

  public String getSASToken(String account, String fileSystem, String path, String operation)
      throws IOException, AccessControlException {

    // Bodo will write the token to this location. This is a temporary directory.
    // See bodo.io.snowflake.SF_AZURE_WRITE_SAS_TOKEN_FILE_LOCATION.
    // which is `os.path.join(bodo.HDFS_CORE_SITE_LOC_DIR.name, "sas_token.txt")`.
    // BODO_HDFS_CORE_SITE_LOC_DIR is set to `bodo.HDFS_CORE_SITE_LOC_DIR.name`
    // in Bodo's __init__.py, so it should be in the
    // enviroment as long as the JVM gets started after import bodo.
    String sasTokenFileLocation = System.getenv("BODO_HDFS_CORE_SITE_LOC_DIR") + "/sas_token.txt";
    // Read the token from file and return it.
    try (FileInputStream inputStream = new FileInputStream(sasTokenFileLocation)) {
      String sasToken = IOUtils.toString(inputStream).strip();
      return sasToken;
    }
  }
}
